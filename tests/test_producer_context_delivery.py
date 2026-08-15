from __future__ import annotations

import contextlib
import io
import json
import os
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
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info["structure_id"] = "s0"
    atoms.info["parent_structure_id"] = "seed-pool:900"
    write(str(dataset), [atoms])
    student = root / "student.yaml"
    student.write_text("kind: mock\ncommittee:\n  seeds: [1]\n", encoding="utf-8")
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({
        "run_id": "producer-context-test",
        "inputs": [str(student), str(dataset)],
        "stages": [{
            "name": "training",
            "command": None,
            "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
            "pydantic_ai": {
                "role": "ml-trainer",
                "action": "train_committee",
                "approval_boundary": "costly_training",
                "idempotency_key": "producer-context-test:training:001",
                "parameters": {
                    "student_config": str(student),
                    "dataset": str(dataset),
                    "output_dir": "{artifacts_dir}/committee",
                    "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
                    "require_lineage": False,
                },
            },
        }],
    }), encoding="utf-8")
    return workflow


def _authoritative(run_dir: Path) -> dict:
    from workflow.controller import RunController
    from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
    c = RunController(run_dir)
    proposal, _ = _proposal_from_stage(c, "training", _stage_config(c, "training"))
    return proposal


class _CaptureRuntime:
    responses: list[str] = []
    tasks: list[dict] = []
    contexts: list[object] = []

    def __init__(self, _responder):
        pass

    def run(self, task, spec, context):
        from runtimes.pydantic_ai.interface import RUNTIME_VERSION, build_invocation
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        self.__class__.tasks.append(task)
        self.__class__.contexts.append(context)
        raw = self.__class__.responses.pop(0)
        toolset = ReadOnlyToolset(context.read_allow_prefixes)
        return build_invocation(task=task, spec=spec, context=context, toolset=toolset,
                                raw_response=raw, usage_source="mock", prompt_tokens=0,
                                completion_tokens=0,
                                runtime_version=f"{RUNTIME_VERSION}+capture")


def _fake_train_committee(student_config, dataset, output_dir, manifest_path):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = out_dir / "seed-1"
    model.write_text("model\n", encoding="utf-8")
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    from workflow.integrity import artifact_digest
    payload = {"schema_version": 1, "models": [{"kind": "mock", "seed": 1,
               "path": str(model.resolve()), "integrity": artifact_digest(model),
               "metadata": {}}]}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class ProducerContextDeliveryTests(unittest.TestCase):
    def _run_training(self, env=None):
        from runtimes.pydantic_ai import cli
        from workflow.controller import RunController
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        run_dir = root / "run"
        RunController.initialize(_workflow(root), run_dir)
        cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training", "--note", "test"])
        _CaptureRuntime.responses = [json.dumps(_authoritative(run_dir))]
        _CaptureRuntime.tasks = []
        _CaptureRuntime.contexts = []
        patch_env = mock.patch.dict(os.environ, env or {}, clear=False)
        patch_runtime = mock.patch("runtimes.pydantic_ai.mock_runtime.MockAgentRuntime", _CaptureRuntime)
        patch_exec = mock.patch("workflow.steps.train_committee", _fake_train_committee)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch_env, patch_runtime, patch_exec, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(["run-stage", "--runtime", "mock", "--run-dir", str(run_dir), "--stage", "training", "--auto-mock-judges"])
        self._last_cli_output = stdout.getvalue() + stderr.getvalue()
        return code, run_dir, RunController(run_dir), list(_CaptureRuntime.tasks), list(_CaptureRuntime.contexts)

    def test_production_producer_receives_inline_primary_evidence_without_tools(self):
        from runtimes.pydantic_ai import cli
        code, _run_dir, c, tasks, contexts = self._run_training()
        self.assertEqual(code, cli.EXIT_SUCCESS)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["inputs"], [])
        self.assertTrue(task["context"]["primary_evidence_inline"])
        packet = task["context"]["producer_evidence_packet"]
        self.assertEqual(packet["packet_kind"], "ProducerEvidencePacket")
        self.assertLessEqual(packet["packet_bytes"], cli.MAX_PRODUCER_EVIDENCE_PACKET_BYTES)
        self.assertFalse(contexts[0].tools_enabled)
        self.assertEqual(contexts[0].read_allow_prefixes, [])
        self.assertEqual(
            c.state["idempotency"]["producer-context-test:training:001:iter1"]["status"], "EXECUTED")

    def test_context_budget_failure_is_local_before_runtime_or_executor(self):
        from runtimes.pydantic_ai import cli
        code, _run_dir, c, tasks, _contexts = self._run_training({"PYDANTIC_AI_CONTEXT_WINDOW_TOKENS": "100"})
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self.assertEqual(tasks, [])
        self.assertIn(cli.PRODUCER_CONTEXT_BUDGET_EXCEEDED, self._last_cli_output)
        self.assertNotIn("producer-context-test:training:001", c.state.get("idempotency", {}))
        self.assertEqual(c.stage("training")["attempts"], 0)
        self.assertEqual(c.stage("training")["status"], "pending")

    def test_output_reserve_is_counted_in_budget(self):
        from orchestration.specs import load_agent_specs
        from runtimes.pydantic_ai import cli
        from runtimes.pydantic_ai.models import RuntimeContext
        from workflow.controller import RunController
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            RunController.initialize(_workflow(root), run_dir)
            c = RunController(run_dir)
            evidence = run_dir / "exchange" / "bounded_evidence" / "training.json"
            from runtimes.pydantic_ai.bounded_evidence import build_bounded_evidence
            build_bounded_evidence([], evidence)
            proposal = _authoritative(run_dir)
            task = cli._producer_task("training", "ml-trainer", evidence, c, proposal)
            spec = load_agent_specs(ROOT / "agent_specs")["ml-trainer"]
            ctx = RuntimeContext(exchange_dir=str(run_dir / "exchange"), repo_root=str(ROOT),
                                 tools_enabled=False)
            base_policy = {"context_window_tokens": 100_000, "output_token_reserve": 100,
                           "prompt_safety_margin_tokens": 200}
            ok, base_diag = cli._producer_context_budget(task, spec, ctx, base_policy)
            self.assertTrue(ok)

            high_reserve_policy = dict(base_policy, output_token_reserve=2000)
            ok, high_diag = cli._producer_context_budget(task, spec, ctx, high_reserve_policy)
            self.assertTrue(ok)
            self.assertGreater(high_diag["estimated_total_tokens"], base_diag["estimated_total_tokens"])
            self.assertEqual(
                high_diag["estimated_total_tokens"] - base_diag["estimated_total_tokens"],
                high_reserve_policy["output_token_reserve"] - base_policy["output_token_reserve"],
            )

            fits_policy = dict(high_reserve_policy,
                               context_window_tokens=high_diag["estimated_total_tokens"] + 16)
            ok, _diag = cli._producer_context_budget(task, spec, ctx, fits_policy)
            self.assertTrue(ok)

            fails_policy = dict(high_reserve_policy,
                                context_window_tokens=high_diag["estimated_total_tokens"] - 1)
            ok, fail_diag = cli._producer_context_budget(task, spec, ctx, fails_policy)
            self.assertFalse(ok)
            self.assertEqual(fail_diag["output_token_reserve"], 2000)

    def test_context_window_is_configurable(self):
        from runtimes.pydantic_ai.cli import producer_context_policy
        with mock.patch.dict(os.environ, {"PYDANTIC_AI_CONTEXT_WINDOW_TOKENS": "12000"}, clear=False):
            policy = producer_context_policy("local-openai", "qwen2.5-7b-instruct")
        self.assertEqual(policy["context_window_tokens"], 12000)
        self.assertEqual(policy["source"], "env")

    def test_budget_failure_is_not_binding_retry(self):
        from runtimes.pydantic_ai import cli
        code, _run_dir, c, tasks, _contexts = self._run_training({"PYDANTIC_AI_CONTEXT_WINDOW_TOKENS": "100"})
        self.assertEqual(code, cli.EXIT_VALIDATION_REJECTED)
        self.assertEqual(tasks, [])
        self.assertNotIn("producer-context-test:training:001", c.state.get("idempotency", {}))


if __name__ == "__main__":
    unittest.main()
