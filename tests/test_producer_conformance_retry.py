from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent


def _workflow(root: Path) -> Path:
    dataset = root / "train.extxyz"
    frames = []
    for i in range(2):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"s{i}"
        a.info["parent_structure_id"] = f"seed-pool:{900 + i}"
        frames.append(a)
    write(str(dataset), frames)
    student = root / "student.yaml"
    student.write_text("kind: mock\ncommittee:\n  seeds: [1]\n")
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "retry-test",
        "inputs": [str(student), str(dataset)],
        "stages": [{
            "name": "training",
            "command": None,
            "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
            "pydantic_ai": {
                "role": "ml-trainer",
                "action": "train_committee",
                "approval_boundary": "costly_training",
                "idempotency_key": "retry-test:training:001",
                "parameters": {
                    "student_config": str(student),
                    "dataset": str(dataset),
                    "output_dir": "{artifacts_dir}/committee",
                    "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                    "require_lineage": False,
                    "optional_note": None,
                    "empty_list": [],
                    "empty_dict": {},
                },
            },
        }],
    }))
    return workflow


def _authoritative(run_dir: Path) -> dict:
    from workflow.controller import RunController
    from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
    c = RunController(run_dir)
    proposal, _ = _proposal_from_stage(c, "training", _stage_config(c, "training"))
    return proposal


class _SequentialRuntime:
    responses: list[str] = []
    tasks: list[dict] = []

    def __init__(self, _responder):
        pass

    def run(self, task, spec, context):
        from runtimes.pydantic_ai.interface import RUNTIME_VERSION, build_invocation
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        self.__class__.tasks.append(task)
        if not self.__class__.responses:
            raise AssertionError("no mock response left")
        raw = self.__class__.responses.pop(0)
        toolset = ReadOnlyToolset(context.read_allow_prefixes)
        return build_invocation(task=task, spec=spec, context=context, toolset=toolset,
                                raw_response=raw, usage_source="mock", prompt_tokens=0,
                                completion_tokens=0,
                                runtime_version=f"{RUNTIME_VERSION}+sequence")


def _run_cli_with_responses(run_dir: Path, responses: list[dict], *, approve=True):
    from runtimes.pydantic_ai import cli
    from workflow.controller import RunController
    calls = {"n": 0}

    def fake_train_committee(student_config, dataset, output_dir, manifest_path):
        calls["n"] += 1
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model = out_dir / "seed-1"
        model.write_text("model\n")
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        from workflow.integrity import artifact_digest
        payload = {"schema_version": 1, "models": [{"kind": "mock", "seed": 1,
                   "path": str(model.resolve()), "integrity": artifact_digest(model),
                   "metadata": {}}]}
        manifest.write_text(json.dumps(payload))
        return payload

    if approve:
        cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training", "--note", "test"])
    _SequentialRuntime.responses = [json.dumps(r) for r in responses]
    _SequentialRuntime.tasks = []
    with mock.patch("runtimes.pydantic_ai.mock_runtime.MockAgentRuntime", _SequentialRuntime), \
         mock.patch("workflow.steps.train_committee", fake_train_committee):
        code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training"])
    return code, calls["n"], list(_SequentialRuntime.tasks), RunController(run_dir)


class ProducerConformanceRetryTests(unittest.TestCase):
    def test_two_invalid_then_valid_executes_once(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from workflow.controller import RunController
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            valid = _authoritative(run_dir)
            invalid1 = json.loads(json.dumps(valid)); invalid1["parameters"].pop("dataset")
            invalid2 = json.loads(json.dumps(valid)); invalid2["parameters"].pop("manifest_path")
            code, executions, tasks, c = _run_cli_with_responses(run_dir, [invalid1, invalid2, valid])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            self.assertEqual(executions, 1)
            self.assertEqual(c.state["idempotency"]["retry-test:training:001"]["status"], "EXECUTED")
            self.assertEqual(len(tasks), 3)
            self.assertIn("producer_retry_feedback", tasks[1]["context"])
            self.assertIn("producer_retry_feedback", tasks[2]["context"])

    def test_three_invalid_attempts_fail_closed(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from workflow.controller import RunController
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            valid = _authoritative(run_dir)
            invalid = json.loads(json.dumps(valid)); invalid["parameters"].pop("dataset")
            code, executions, tasks, c = _run_cli_with_responses(run_dir, [invalid, invalid, invalid])
            self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
            self.assertEqual(executions, 0)
            self.assertNotIn("retry-test:training:001", c.state.get("idempotency", {}))
            self.assertEqual(c.stage("training")["status"], "pending")
            self.assertEqual(len(tasks), 3)

    def test_changed_parameter_then_corrected_still_strict(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from workflow.controller import RunController
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            valid = _authoritative(run_dir)
            invalid = json.loads(json.dumps(valid)); invalid["parameters"]["dataset"] = str(root / "other.extxyz")
            code, executions, _tasks, c = _run_cli_with_responses(run_dir, [invalid, valid])
            self.assertEqual(code, cli.EXIT_SUCCESS)
            self.assertEqual(executions, 1)
            self.assertEqual(c.stage("training")["status"], "completed")

    def test_extra_and_missing_false_null_empty_are_rejected_by_binding(self):
        from runtimes.pydantic_ai.cli import _proposal_binding_validator
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            c = RunController(run_dir)
            valid = _authoritative(run_dir)
            validator = _proposal_binding_validator(valid, c)
            extra = json.loads(json.dumps(valid)); extra["parameters"]["extra"] = "x"
            self.assertFalse(validator(extra)[0])
            for key in ["require_lineage", "optional_note", "empty_list", "empty_dict"]:
                candidate = json.loads(json.dumps(valid))
                candidate["parameters"].pop(key)
                ok, message = validator(candidate)
                self.assertFalse(ok, key)
                self.assertIn("parameters", message)

    def test_unapproved_conforming_proposal_stops_at_approval_without_executor(self):
        from runtimes.pydantic_ai import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from workflow.controller import RunController
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            valid = _authoritative(run_dir)
            code, executions, tasks, c = _run_cli_with_responses(run_dir, [valid], approve=False)
            self.assertEqual(code, cli.EXIT_APPROVAL_REQUIRED)
            self.assertEqual(executions, 0)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(c.stage("training")["status"], "pending")
            self.assertFalse(c.state.get("action_approvals"))

    def test_protected_reference_and_judge_paths_are_unchanged(self):
        from runtimes.pydantic_ai.cli import _protection_consuming_action, MAX_PRODUCER_GENERATION_ATTEMPTS
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        self.assertEqual(MAX_PRODUCER_GENERATION_ATTEMPTS, 3)
        self.assertTrue(_protection_consuming_action("acquire_structures"))
        self.assertTrue(_protection_consuming_action("label_with_teacher"))
        self.assertTrue(_protection_consuming_action("train_committee"))
        self.assertFalse(_protection_consuming_action("validate_teacher_reference"))
        self.assertNotIn("judge", ROLE_ALLOWED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
