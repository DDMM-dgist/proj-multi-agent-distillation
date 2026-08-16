"""Generic campaign progress/observability event layer (Phase 7).

Emits a single, generic event vocabulary describing OBSERVABLE workflow state -- campaign
start/resume, stage progress, PydanticAI role-invocation start/completion, typed-proposal
accept/reject, approval-boundary status, executor start/completion/pending/progress, artifact
registration, judge invocation/result, gate result, recovery diagnosis/proposal/approval state,
resource/external-job pause, and terminal campaign outcome. Never a specific stage name, agent
role vocabulary, or domain concept of its own: every field is either generic metadata (timestamps,
stage index/total) or a value already produced by the existing Controller/dispatch/production-
router layers this module only observes -- it changes NO Controller or dispatch semantics.

Never emits private model chain-of-thought: only typed/result summaries and workflow state that
were already the accepted, structured output of a role dispatch (never raw provider text).

Two independent concerns:
  - durable persistence: every event is always appended to ``<run_dir>/campaign_events.jsonl``,
    regardless of console settings -- this is unconditional, not opt-in.
  - console rendering: human-readable by default (production behavior), ``--json-events`` streams
    the SAME event objects as JSON lines instead, ``--quiet`` suppresses console output entirely
    (persistence is unaffected by either flag).

UTF-8 safety: the JSONL file is always opened with an explicit ``encoding="utf-8"`` (never the
locale-dependent default), and console writes go through ``_write_safely``, which falls back to a
lossy-but-non-crashing re-encode if the console stream's own encoding (e.g. ASCII under a ``C``
locale) cannot represent a character -- so unicode content (an em-dash in a rationale string, etc.)
can never crash campaign execution or event persistence.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

EVENT_SCHEMA_VERSION = 1
EVENTS_FILENAME = "campaign_events.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_safely(stream, text: str) -> None:
    """Write ``text`` to ``stream`` without ever raising on a character the stream's own encoding
    cannot represent (e.g. a real terminal under ``LANG=C``/ASCII). Falls back to a lossy
    re-encode/decode round-trip through the stream's declared encoding (or utf-8) with
    ``errors="replace"`` -- observability output must never be the reason a campaign crashes."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe)
    if hasattr(stream, "flush"):
        stream.flush()


_CAMPAIGN_LIFECYCLE_START_EVENTS = ("campaign_started", "campaign_resumed")


def campaign_previously_executed(run_dir) -> bool:
    """True iff a PRIOR ``run-campaign`` invocation actually executed against this ``run_dir`` --
    i.e. the durable event log already contains a campaign_started/campaign_resumed lifecycle
    event of its own. Deliberately NOT ``events_path.exists()``: other commands (``approve``,
    ``approve-recovery``) write their OWN events (e.g. ``approval_granted``) to this same durable
    log via their own ``CampaignEventEmitter``, so the file can exist before ``run-campaign`` has
    ever executed once -- file existence alone is not evidence of a prior campaign execution."""
    path = Path(run_dir) / EVENTS_FILENAME
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") in _CAMPAIGN_LIFECYCLE_START_EVENTS:
                return True
    return False


def terminal_class(outcome: str) -> str:
    """Bucket any campaign outcome string into the tri-state COMPLETED/FAILED/PAUSED vocabulary --
    every non-COMPLETED, non-FAILED outcome (waiting for human approval, waiting for recovery
    evidence, resource-blocked, recovery-required, recovery-execution-unverified) is a pause: the
    campaign stopped short of a terminal state and the identical command resumes it later."""
    if outcome == "COMPLETED":
        return "COMPLETED"
    if outcome == "FAILED":
        return "FAILED"
    return "PAUSED"


def stage_progress_fields(controller, stage_name: Optional[str] = None) -> dict:
    """Generic ``{"stage", "stage_index", "stage_total"}`` derived entirely from the Controller's
    own declared stage list -- no stage name, count, or domain concept is assumed by this module.
    With no ``stage_name``, reports the first stage whose gate has not resolved as PASS or
    NOT_APPLICABLE (mirrors ``cli._next_eligible_stage`` without importing it, to avoid a
    cli<->events import cycle)."""
    stages = controller.state.get("stages", [])
    total = len(stages)
    if stage_name is None:
        for index, stage in enumerate(stages, 1):
            if stage.get("gate") not in ("PASS", "NOT_APPLICABLE"):
                return {"stage": stage["name"], "stage_index": index, "stage_total": total}
        return {"stage": None, "stage_index": None, "stage_total": total}
    for index, stage in enumerate(stages, 1):
        if stage["name"] == stage_name:
            return {"stage": stage_name, "stage_index": index, "stage_total": total}
    return {"stage": stage_name, "stage_index": None, "stage_total": total}


class CampaignEventEmitter:
    """Appends generic campaign events to a durable JSONL log and, by default, prints a concise
    human-readable progress line for each one. Construct once per ``run-campaign``/``run-stage``
    invocation and thread the SAME instance through every layer that observes campaign progress."""

    def __init__(self, run_dir, *, run_id: Optional[str] = None, quiet: bool = False,
                json_events: bool = False, stream=None):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / EVENTS_FILENAME
        self.run_id = run_id
        self.quiet = quiet
        self.json_events = json_events
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, event: str, *, stage: Optional[str] = None, stage_index: Optional[int] = None,
            stage_total: Optional[int] = None, role: Optional[str] = None,
            action: Optional[str] = None, detail: Optional[dict] = None) -> dict:
        record = {"schema_version": EVENT_SCHEMA_VERSION, "ts": _now(), "event": event}
        if self.run_id:
            record["run_id"] = self.run_id
        for key, value in (("stage", stage), ("stage_index", stage_index),
                           ("stage_total", stage_total), ("role", role), ("action", action)):
            if value is not None:
                record[key] = value
        if detail:
            record["detail"] = detail
        self._persist(record)
        self._write_console(record)
        return record

    def _persist(self, record: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        # Always explicit utf-8: never the locale-dependent default text-mode encoding, so a
        # non-ASCII character (e.g. an em-dash in a rationale) can never crash persistence even
        # under an ASCII/"C"-locale process environment.
        with open(self.events_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def _write_console(self, record: dict) -> None:
        if self.quiet:
            return
        if self.json_events:
            _write_safely(self.stream, json.dumps(record, ensure_ascii=False) + "\n")
            return
        _write_safely(self.stream, _render_human(record) + "\n")


def _render_human(record: dict) -> str:
    ts = record["ts"]
    time_part = ts.split("T", 1)[1].split("+", 1)[0].split(".", 1)[0] if "T" in ts else ts
    parts = [f"[{time_part}]", record["event"]]
    stage = record.get("stage")
    if stage:
        progress = ""
        if record.get("stage_index") and record.get("stage_total"):
            progress = f" ({record['stage_index']}/{record['stage_total']})"
        parts.append(f"stage={stage}{progress}")
    if record.get("role"):
        parts.append(f"role={record['role']}")
    if record.get("action"):
        parts.append(f"action={record['action']}")
    detail = record.get("detail") or {}
    for key in ("status", "decision", "accepted", "outcome", "verdict"):
        if key in detail:
            parts.append(f"{key}={detail[key]}")
    return " ".join(parts)
