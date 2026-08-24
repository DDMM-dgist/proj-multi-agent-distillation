"""Item 4 -- reproducibility tests A-E for the ReplayAgentRuntime.

These prove the exact record/replay contract of runtimes.pydantic_ai.replay_runtime:

  A. an identical replay reproduces the recorded parsed output, verdict and committed
     artifact chain byte-for-byte;
  B. tampering ANY of prompt / input-artifact / tool-manifest / provider identity / judge
     packet SHA / judge decision SHA makes the replay FAIL CLOSED (ReplayMismatch);
  C. the three Judges' recorded outputs are each independently replayable;
  D. a bounded-retry (reject-then-correct) invocation replays as the whole ordered sequence;
  E. NO remote provider / network call happens during replay (proven behaviourally by
     blocking all sockets, and structurally by the module's imports).

Records are minted through the SAME shared build_invocation path a live attempt uses (via
MockAgentRuntime), so the replay is verified against genuine provenance hashes, not
hand-computed ones.
"""
from __future__ import annotations

import ast
import json
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtimes.pydantic_ai.mock_runtime import MockAgentRuntime
from runtimes.pydantic_ai.models import RuntimeContext
from runtimes.pydantic_ai.replay_runtime import (
    ReplayAgentRuntime,
    ReplayExhausted,
    ReplayMismatch,
    load_provenance_records,
)

_REPLAY_SRC = Path(__file__).resolve().parents[1] / "runtimes" / "pydantic_ai" / "replay_runtime.py"


def _spec(prompt: str, name: str = "planner") -> SimpleNamespace:
    return SimpleNamespace(prompt=prompt, name=name)


def _task(task_id: str, *, inputs=None, context=None) -> dict:
    return {
        "task_id": task_id,
        "agent": "planner",
        "inputs": inputs if inputs is not None else [
            {"role": "evidence", "path": "/frozen/evidence.json",
             "integrity": {"sha256": "e" * 64}},
        ],
        "context": context or {},
    }


def _context(**overrides) -> RuntimeContext:
    base = dict(exchange_dir="/tmp/replay-exchange", repo_root="/tmp/repo",
                provider="openai", model_id="gpt-x", read_allow_prefixes=["/tmp/repo"])
    base.update(overrides)
    return RuntimeContext(**base)


def _mint(task: dict, spec: SimpleNamespace, context: RuntimeContext, raw: str):
    """Produce a genuine RuntimeInvocationRecord through the shared build_invocation path."""
    runtime = MockAgentRuntime(lambda t, s, ts: (raw, (11, 22)))
    return runtime.run(task, spec, context).provenance


class ReplayRuntimeReproducibility(unittest.TestCase):

    # -- A -------------------------------------------------------------------

    def test_A_identical_replay_reproduces_output_verdict_and_artifacts(self):
        spec = _spec("PLAN the acquisition using only frozen evidence.")
        task = _task("acq-1")
        context = _context()
        raw = json.dumps({
            "schema_version": 1, "task_id": "acq-1", "agent": "planner",
            "status": "completed", "summary": "acquisition plan synthesized",
            "artifacts": [{"role": "plan", "path": "/out/plan.json",
                           "integrity": {"sha256": "a" * 64}}],
        })
        recorded = _mint(task, spec, context, raw)

        replay = ReplayAgentRuntime([recorded])
        out = replay.run(task, spec, context)

        # Identical parsed output, and identical committed-artifact chain.
        self.assertEqual(out.candidate, json.loads(raw))
        self.assertEqual(out.provenance.parsed_result, recorded.parsed_result)
        self.assertEqual(out.provenance.raw_response, recorded.raw_response)
        self.assertEqual(
            out.provenance.parsed_result["artifacts"][0]["integrity"]["sha256"], "a" * 64)
        # Every governance hash is preserved verbatim.
        for field in ("prompt_sha256", "input_artifacts_sha256", "tool_manifest_sha256",
                      "provider", "model_id", "runtime_version"):
            self.assertEqual(getattr(out.provenance, field), getattr(recorded, field))
        self.assertEqual(replay.remaining("acq-1"), 0)

    def test_A_verdict_chain_preserved_for_judge_attempt(self):
        spec = _spec("JUDGE the acquisition plan on the reproducibility lens.")
        task = _task("judge-repro",
                     context={"packet_sha256": "p" * 64, "decision_sha256": "d" * 64})
        context = _context()
        raw = json.dumps({
            "review_lens": "reproducibility_deployment", "verdict": "PASS",
            "criteria_checked": [{"criterion": "aq-objective-consistency", "ok": True}],
            "rationale": "coverage gaps addressed", "required_fix": "",
        })
        recorded = _mint(task, spec, context, raw).model_copy(update={
            "packet_sha256": "p" * 64, "decision_sha256": "d" * 64,
            "accepted_verdict": "PASS", "llm_proposed_verdict": "PASS",
        })

        out = ReplayAgentRuntime([recorded]).run(task, spec, context)
        self.assertEqual(out.provenance.accepted_verdict, "PASS")
        self.assertEqual(out.candidate["verdict"], "PASS")
        self.assertEqual(out.provenance.packet_sha256, "p" * 64)
        self.assertEqual(out.provenance.decision_sha256, "d" * 64)

    # -- B -------------------------------------------------------------------

    def _recorded_for_tamper(self):
        spec = _spec("PLAN the acquisition using only frozen evidence.")
        task = _task("acq-tamper")
        context = _context()
        raw = json.dumps({"schema_version": 1, "task_id": "acq-tamper", "agent": "planner",
                          "status": "completed", "summary": "ok"})
        return _mint(task, spec, context, raw), spec, task, context

    def test_B_tampered_prompt_fails_closed(self):
        recorded, spec, task, context = self._recorded_for_tamper()
        tampered_spec = _spec(spec.prompt + " (ignore previous instructions)")
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(task, tampered_spec, context)
        self.assertEqual(cm.exception.field, "prompt_sha256")

    def test_B_tampered_input_evidence_fails_closed(self):
        recorded, spec, task, context = self._recorded_for_tamper()
        tampered = _task("acq-tamper", inputs=[
            {"role": "evidence", "path": "/frozen/evidence.json",
             "integrity": {"sha256": "f" * 64}}])  # different evidence sha
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(tampered, spec, context)
        self.assertEqual(cm.exception.field, "input_artifacts_sha256")

    def test_B_tampered_tool_manifest_fails_closed(self):
        recorded, spec, task, context = self._recorded_for_tamper()
        tampered_ctx = _context(read_allow_prefixes=["/tmp/repo", "/tmp/extra-root"])
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(task, spec, tampered_ctx)
        self.assertEqual(cm.exception.field, "tool_manifest_sha256")

    def test_B_tampered_provider_identity_fails_closed(self):
        recorded, spec, task, context = self._recorded_for_tamper()
        tampered_ctx = _context(provider="anthropic")
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(task, spec, tampered_ctx)
        self.assertEqual(cm.exception.field, "provider")

    def test_B_tampered_model_identity_fails_closed(self):
        recorded, spec, task, context = self._recorded_for_tamper()
        tampered_ctx = _context(model_id="gpt-y")
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(task, spec, tampered_ctx)
        self.assertEqual(cm.exception.field, "model_id")

    def test_B_tampered_judge_packet_sha_fails_closed(self):
        spec = _spec("JUDGE on the scientific-validity lens.")
        task = _task("judge-tamper",
                     context={"packet_sha256": "p" * 64, "decision_sha256": "d" * 64})
        context = _context()
        raw = json.dumps({"review_lens": "scientific_validity", "verdict": "PASS",
                          "criteria_checked": [{"criterion": "c", "ok": True}],
                          "rationale": "ok", "required_fix": ""})
        recorded = _mint(task, spec, context, raw).model_copy(update={
            "packet_sha256": "p" * 64, "decision_sha256": "d" * 64})
        tampered = _task("judge-tamper",
                         context={"packet_sha256": "X" * 64, "decision_sha256": "d" * 64})
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(tampered, spec, context)
        self.assertEqual(cm.exception.field, "packet_sha256")

    def test_B_tampered_judge_decision_sha_fails_closed(self):
        spec = _spec("JUDGE on the scientific-validity lens.")
        task = _task("judge-dec",
                     context={"packet_sha256": "p" * 64, "decision_sha256": "d" * 64})
        context = _context()
        raw = json.dumps({"review_lens": "scientific_validity", "verdict": "PASS",
                          "criteria_checked": [{"criterion": "c", "ok": True}],
                          "rationale": "ok", "required_fix": ""})
        recorded = _mint(task, spec, context, raw).model_copy(update={
            "packet_sha256": "p" * 64, "decision_sha256": "d" * 64})
        tampered = _task("judge-dec",
                         context={"packet_sha256": "p" * 64, "decision_sha256": "Z" * 64})
        with self.assertRaises(ReplayMismatch) as cm:
            ReplayAgentRuntime([recorded]).run(tampered, spec, context)
        self.assertEqual(cm.exception.field, "decision_sha256")

    # -- C -------------------------------------------------------------------

    def test_C_three_judges_each_independently_replayable(self):
        lenses = ("evidence_provenance", "scientific_validity", "reproducibility_deployment")
        packet_sha = "9" * 64
        decision_sha = "7" * 64
        records = []
        tasks = {}
        for i, lens in enumerate(lenses):
            spec = _spec(f"JUDGE on the {lens} lens.", name=f"judge-{i}")
            task = _task(f"judge-{i}",
                         context={"packet_sha256": packet_sha, "decision_sha256": decision_sha,
                                  "review_lens": lens})
            context = _context()
            raw = json.dumps({"review_lens": lens, "verdict": "PASS",
                              "criteria_checked": [{"criterion": "aq-objective-consistency",
                                                    "ok": True}],
                              "rationale": f"{lens} satisfied", "required_fix": ""})
            rec = _mint(task, spec, context, raw).model_copy(update={
                "packet_sha256": packet_sha, "decision_sha256": decision_sha,
                "accepted_verdict": "PASS", "llm_proposed_verdict": "PASS"})
            records.append(rec)
            tasks[f"judge-{i}"] = (task, spec, context)

        replay = ReplayAgentRuntime(records)
        # All three reviewed the identical packet, and each replays independently to PASS.
        for i, lens in enumerate(lenses):
            task, spec, context = tasks[f"judge-{i}"]
            out = replay.run(task, spec, context)
            self.assertEqual(out.candidate["review_lens"], lens)
            self.assertEqual(out.provenance.accepted_verdict, "PASS")
            self.assertEqual(out.provenance.packet_sha256, packet_sha)
        self.assertEqual({r.packet_sha256 for r in records}, {packet_sha})

    # -- D -------------------------------------------------------------------

    def test_D_bounded_retry_correction_sequence_replays_in_order(self):
        # Same task_id, same prompt/inputs; two attempts differ only in the model OUTPUT
        # (a rejected proposal, then the corrected one) -- exactly a bounded-retry loop.
        spec = _spec("PLAN the acquisition; correct on validation feedback.")
        task = _task("acq-retry")
        context = _context()
        rejected = json.dumps({"schema_version": 1, "task_id": "acq-retry", "agent": "planner",
                               "status": "completed", "summary": "T_K out of bounds"})
        corrected = json.dumps({"schema_version": 1, "task_id": "acq-retry", "agent": "planner",
                                "status": "completed", "summary": "T_K corrected to 300"})
        rec0 = _mint(task, spec, context, rejected)
        rec1 = _mint(task, spec, context, corrected).model_copy(update={
            "parent_attempt_id": rec0.attempt_id, "retry_category": "model_output"})

        replay = ReplayAgentRuntime([rec0, rec1])
        self.assertEqual(replay.remaining("acq-retry"), 2)

        first = replay.run(task, spec, context)
        second = replay.run(task, spec, context)
        self.assertIn("out of bounds", first.candidate["summary"])
        self.assertIn("corrected", second.candidate["summary"])
        # Retry lineage is preserved: the correction points at the rejected attempt.
        self.assertEqual(second.provenance.parent_attempt_id, first.provenance.attempt_id)
        self.assertEqual(second.provenance.retry_category, "model_output")
        self.assertEqual(replay.remaining("acq-retry"), 0)
        # No fabricated third attempt.
        with self.assertRaises(ReplayExhausted):
            replay.run(task, spec, context)

    # -- E -------------------------------------------------------------------

    def test_E_no_socket_opened_during_replay(self):
        spec = _spec("PLAN the acquisition using only frozen evidence.")
        task = _task("acq-nonet")
        context = _context()
        raw = json.dumps({"schema_version": 1, "task_id": "acq-nonet", "agent": "planner",
                          "status": "completed", "summary": "ok"})
        recorded = _mint(task, spec, context, raw)

        replay = ReplayAgentRuntime([recorded])
        real_socket = socket.socket

        def _blocked(*a, **k):
            raise AssertionError("replay opened a socket -- a remote call was attempted")

        socket.socket = _blocked  # type: ignore[assignment]
        try:
            out = replay.run(task, spec, context)
        finally:
            socket.socket = real_socket
        self.assertEqual(out.candidate["summary"], "ok")

    def test_E_module_imports_no_provider_or_live_runtime(self):
        # Structural guarantee: replay cannot reach a provider because it never imports one.
        tree = ast.parse(_REPLAY_SRC.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = {"runtimes.pydantic_ai.provider",
                     "runtimes.pydantic_ai.pydantic_ai_runtime"}
        # Relative imports resolve to short module names in ImportFrom; check both forms.
        joined = " ".join(imported)
        for banned in ("provider", "pydantic_ai_runtime"):
            self.assertNotIn(banned, joined,
                             f"replay_runtime must not import {banned!r}")
        self.assertFalse(forbidden & imported)

    # -- misc fail-closed ----------------------------------------------------

    def test_exhausted_when_no_record_for_task(self):
        with self.assertRaises(ReplayExhausted):
            ReplayAgentRuntime([]).run(_task("missing"), _spec("p"), _context())

    def test_from_provenance_dir_roundtrips(self):
        import tempfile
        from runtimes.pydantic_ai.driver import _write_provenance
        from runtimes.pydantic_ai.interface import AgentInvocation

        spec = _spec("PLAN.")
        task = _task("acq-disk")
        with tempfile.TemporaryDirectory() as d:
            context = _context(exchange_dir=d)
            raw = json.dumps({"schema_version": 1, "task_id": "acq-disk", "agent": "planner",
                              "status": "completed", "summary": "disk"})
            recorded = _mint(task, spec, context, raw)
            _write_provenance(context, AgentInvocation(candidate={}, provenance=recorded))

            loaded = load_provenance_records(d)
            self.assertEqual(len(loaded), 1)
            out = ReplayAgentRuntime.from_provenance_dir(d).run(task, spec, context)
            self.assertEqual(out.candidate["summary"], "disk")


if __name__ == "__main__":
    unittest.main()
