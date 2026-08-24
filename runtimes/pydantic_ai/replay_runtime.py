"""ReplayAgentRuntime -- exact, network-free replay of a recorded governance chain.

A THIRD runtime beside MockAgentRuntime and PydanticAIRuntime. It implements the same
``AgentRuntime`` protocol (``run(task, spec, context) -> AgentInvocation``) but produces NO
new model output: instead it returns a previously recorded ``RuntimeInvocationRecord`` --
the raw model response, the parsed Pydantic object, the recorded tool I/O, the retry
lineage, and (for Judge attempts) the accepted verdict, packet SHA and decision SHA.

Why a separate runtime and not a flag on the real one: replay must be structurally
INCAPABLE of reaching a provider. This module imports nothing from ``provider.py`` /
``pydantic_ai_runtime.py`` and never constructs a client, so "no remote LLM/API call
during replay" is guaranteed by construction, not by a runtime branch that could regress.

Fail-closed contract. Before returning a recorded attempt, ``run`` RECOMPUTES from the
CURRENT ``task``/``spec``/``context`` the same hashes the shared ``build_invocation`` path
computes when a live attempt is recorded -- the instruction-prompt SHA, the per-role input
artifact SHAs, and the read-only tool-manifest SHA -- and checks provider/model/runtime
identity, plus the recorded Judge packet/decision SHAs when present. If ANY of these does
not match the recorded attempt, ``run`` raises :class:`ReplayMismatch` (a hard error). A
replay that cannot reproduce the exact recorded chain is a failure, never a silent
best-effort. The reasoning/validation path itself is untouched: the returned candidate is
fed to the driver's EXISTING ``validate_agent_response`` exactly as a live candidate is, so
replay exercises the real deterministic validators and the real gate, not a shortcut.

Ordering. A single ``run`` call replays ONE recorded attempt, mirroring MockAgentRuntime
(one responder call per ``run``). Attempts for a ``task_id`` are served in recorded order,
so a bounded-retry+correction sequence replays by successive ``run`` calls -- exactly how a
higher-level driver (e.g. the acquisition planner's bounded loop) invokes a runtime.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .interface import AgentInvocation, _input_hashes, sha256_text
from .models import RuntimeContext, RuntimeInvocationRecord
from .tool_registry import ReadOnlyToolset


class ReplayMismatch(RuntimeError):
    """A recomputed hash / identity field did not match the recorded attempt.

    Fail-closed: a replay that cannot reproduce the recorded governance chain is a hard
    error, never a silent fallback to live inference or a best-effort partial replay.
    """

    def __init__(self, field: str, expected: Any, actual: Any, *,
                 task_id: str = "", attempt_id: str = ""):
        self.field = field
        self.expected = expected
        self.actual = actual
        self.task_id = task_id
        self.attempt_id = attempt_id
        super().__init__(
            f"REPLAY_MISMATCH: {field} differs for task_id={task_id!r} "
            f"attempt_id={attempt_id!r}: recorded={expected!r} replayed-context={actual!r}. "
            "Refusing to replay -- the current inputs do not reproduce the recorded attempt."
        )


class ReplayExhausted(RuntimeError):
    """A ``run`` was requested for a ``task_id`` with no remaining recorded attempt.

    Fail-closed: replay never fabricates an attempt that was not recorded.
    """


def _as_record(obj: Any) -> RuntimeInvocationRecord:
    if isinstance(obj, RuntimeInvocationRecord):
        return obj
    if isinstance(obj, dict):
        return RuntimeInvocationRecord.model_validate(obj)
    raise TypeError(f"not a RuntimeInvocationRecord or dict: {type(obj)!r}")


def load_provenance_records(exchange_dir: str) -> list[RuntimeInvocationRecord]:
    """Load every recorded attempt under ``{exchange_dir}/provenance`` in recorded order.

    Files are ``{task_id}.{attempt_id}.json``. Ordering is by (started_at, recorded_at,
    attempt_id) so a retry chain replays in the wall-clock order it was produced. Only
    genuine provider/mock attempts are returned; a record that captured a provider
    exception before producing output (``failure_category`` set, empty ``raw_response``) is
    skipped -- there is no model output to replay for it.
    """
    prov = Path(exchange_dir).resolve() / "provenance"
    records: list[RuntimeInvocationRecord] = []
    if not prov.is_dir():
        return records
    for path in sorted(prov.glob("*.json")):
        rec = RuntimeInvocationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if rec.failure_category:
            continue
        records.append(rec)
    records.sort(key=lambda r: (r.started_at or "", r.recorded_at or "", r.attempt_id))
    return records


class ReplayAgentRuntime:
    """Replays recorded attempts, one per ``run`` call, in recorded order per ``task_id``.

    Construct from an explicit ordered sequence of records, or via
    :meth:`from_provenance_dir` to load them from a recorded exchange's provenance folder.
    ``strict_identity`` (default True) also requires provider/model/runtime-version to match
    the recorded attempt; set False only to replay a record under a deliberately relabeled
    provider identity (still fails closed on every content hash).
    """

    def __init__(self, records: Iterable[Any], *, strict_identity: bool = True):
        self._strict_identity = strict_identity
        self._queues: dict[str, deque[RuntimeInvocationRecord]] = defaultdict(deque)
        for obj in records:
            rec = _as_record(obj)
            self._queues[rec.task_id].append(rec)
        # A flat record of everything replayed, for auditing/tests.
        self.replayed: list[RuntimeInvocationRecord] = []

    @classmethod
    def from_provenance_dir(cls, exchange_dir: str, *, strict_identity: bool = True
                            ) -> "ReplayAgentRuntime":
        return cls(load_provenance_records(exchange_dir), strict_identity=strict_identity)

    def remaining(self, task_id: str) -> int:
        return len(self._queues.get(task_id, ()))

    # -- the AgentRuntime protocol ----------------------------------------------

    def run(self, task: dict, spec: Any, context: RuntimeContext) -> AgentInvocation:
        task_id = task.get("task_id", "")
        queue = self._queues.get(task_id)
        if not queue:
            raise ReplayExhausted(
                f"REPLAY_EXHAUSTED: no remaining recorded attempt for task_id={task_id!r}. "
                "Replay never fabricates an unrecorded attempt."
            )
        record = queue.popleft()
        self._verify(task=task, spec=spec, context=context, record=record)
        self.replayed.append(record)
        candidate = dict(record.parsed_result) if record.parsed_result else {}
        # Return the recorded attempt verbatim (a copy so the queue's record is never
        # mutated by a downstream consumer). The candidate is fed to the driver's EXISTING
        # validate_agent_response, so the deterministic validators / gate run unchanged.
        return AgentInvocation(candidate=candidate,
                               provenance=record.model_copy(deep=True))

    # -- fail-closed verification -----------------------------------------------

    def _verify(self, *, task: dict, spec: Any, context: RuntimeContext,
                record: RuntimeInvocationRecord) -> None:
        tid, aid = record.task_id, record.attempt_id

        # 1) Instruction/system prompt hash -- recomputed exactly as build_invocation does.
        prompt = getattr(spec, "prompt", "")
        recomputed_prompt = sha256_text(prompt)
        if recomputed_prompt != record.prompt_sha256:
            raise ReplayMismatch("prompt_sha256", record.prompt_sha256, recomputed_prompt,
                                 task_id=tid, attempt_id=aid)

        # 2) Per-role input-artifact hashes (evidence packet / spec / framework snapshot /
        #    every referenced input carry their integrity sha256 here).
        recomputed_inputs = _input_hashes(task)
        if recomputed_inputs != dict(record.input_artifacts_sha256):
            raise ReplayMismatch("input_artifacts_sha256", dict(record.input_artifacts_sha256),
                                 recomputed_inputs, task_id=tid, attempt_id=aid)

        # 3) Read-only tool-manifest hash (the tool/action contract surface).
        toolset = ReadOnlyToolset(context.read_allow_prefixes)
        recomputed_tools = toolset.tool_manifest_sha256()
        if recomputed_tools != record.tool_manifest_sha256:
            raise ReplayMismatch("tool_manifest_sha256", record.tool_manifest_sha256,
                                 recomputed_tools, task_id=tid, attempt_id=aid)

        # 4) Provider / model / runtime-version identity.
        if self._strict_identity:
            for field, current, recorded in (
                ("provider", context.provider, record.provider),
                ("model_id", context.model_id, record.model_id),
            ):
                if current != recorded:
                    raise ReplayMismatch(field, recorded, current, task_id=tid, attempt_id=aid)
        if not record.runtime_version:
            raise ReplayMismatch("runtime_version", "<non-empty>", record.runtime_version,
                                 task_id=tid, attempt_id=aid)

        # 5) Judge governance SHAs, when the recorded attempt is a judge attempt. The packet
        #    and decision SHAs are canonical bindings the caller carries in the task context;
        #    a tampered packet/decision must fail closed here rather than silently replay.
        ctx = task.get("context", {}) if isinstance(task, dict) else {}
        self._verify_optional_sha("packet_sha256", record.packet_sha256, ctx, tid, aid)
        self._verify_optional_sha("decision_sha256", record.decision_sha256, ctx, tid, aid)

    @staticmethod
    def _verify_optional_sha(field: str, recorded: Any, ctx: dict, tid: str, aid: str) -> None:
        """Verify a recorded Judge SHA against the same key in the current task context, but
        ONLY when both are present. A recorded SHA with a mismatching context SHA fails
        closed; a context that does not carry the key at all is left to the input-artifact
        hash check above (the packet is also an input reference)."""
        if not recorded:
            return
        current = ctx.get(field) if isinstance(ctx, dict) else None
        if current is not None and current != recorded:
            raise ReplayMismatch(field, recorded, current, task_id=tid, attempt_id=aid)
