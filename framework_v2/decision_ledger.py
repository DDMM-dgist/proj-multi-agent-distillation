"""Framework V2 — DecisionLedger (Section 16).

Append-only, per-decision, machine-readable record so that the framework can
answer the "why this X?" audit questions from Section 16 without terminal-log
archaeology.

Storage model
-------------

Ledger data lives in a run-scoped directory, typically::

    <run_dir>/framework_v2/decision_ledger/
        decisions.jsonl        # one JSON object per ScientificDecisionRecord
        facts.jsonl            # one JSON object per DeterministicFact
        contradictions.jsonl   # one JSON object per JudgeContradiction

All three are append-only JSONL. They are never rewritten -- superseded
decisions get a new record with a ``supersedes`` field, preserving history.

The ledger is safe to read concurrently but has a single writer per file
per process. Callers must serialize writes at their level (the Controller
does this via its single-thread state loop).

Validation rules
----------------

* ``ScientificDecisionRecord.decision_id`` must be unique within
  ``decisions.jsonl``.
* Every ``deterministic_facts`` entry in a record must exist in
  ``facts.jsonl`` before the record is appended (fail closed).
* A ``ScientificDecisionRecord`` whose ``provenance_class`` is
  ``LEGACY_REUSED`` or ``TOOL_DEFAULT`` must have a non-empty
  ``rationale`` -- enforced by ``append_decision`` (Section 9 &
  Section 16 combined).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from framework_v2.contracts import (
    ProvenanceClass,
    ScientificDecisionRecord,
)
from framework_v2.facts import (
    DeterministicFact,
    JudgeContradiction,
)


DECISIONS_FILENAME = "decisions.jsonl"
FACTS_FILENAME = "facts.jsonl"
CONTRADICTIONS_FILENAME = "contradictions.jsonl"


class DecisionLedgerError(RuntimeError):
    """Fail-closed error raised on invalid ledger operation."""


class DecisionLedger:
    """A run-scoped append-only ledger.

    ``root`` is the directory that holds the three JSONL files (created
    on demand). The ledger is not thread-safe; the Controller must
    serialize calls.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.root / DECISIONS_FILENAME
        self.facts_path = self.root / FACTS_FILENAME
        self.contradictions_path = self.root / CONTRADICTIONS_FILENAME

    # ---- reads ---------------------------------------------------------
    def _read_jsonl(self, path: Path) -> Iterator[dict]:
        if not path.exists():
            return iter(())
        def _gen():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return _gen()

    def iter_decisions(self) -> Iterator[ScientificDecisionRecord]:
        for row in self._read_jsonl(self.decisions_path):
            yield ScientificDecisionRecord.model_validate(row)

    def iter_facts(self) -> Iterator[DeterministicFact]:
        for row in self._read_jsonl(self.facts_path):
            yield DeterministicFact.model_validate(row)

    def iter_contradictions(self) -> Iterator[JudgeContradiction]:
        for row in self._read_jsonl(self.contradictions_path):
            yield JudgeContradiction.model_validate(row)

    def facts_by_id(self, ids: Iterable[str]) -> dict[str, DeterministicFact]:
        want = set(ids)
        found: dict[str, DeterministicFact] = {}
        for f in self.iter_facts():
            if f.fact_id in want:
                found[f.fact_id] = f
                if len(found) == len(want):
                    break
        return found

    def known_decision_ids(self) -> set[str]:
        return {d.decision_id for d in self.iter_decisions()}

    def known_fact_ids(self) -> set[str]:
        return {f.fact_id for f in self.iter_facts()}

    # ---- appends -------------------------------------------------------
    def _append_jsonl(self, path: Path, record: dict) -> None:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    def append_fact(self, fact: DeterministicFact) -> None:
        if fact.fact_id in self.known_fact_ids():
            raise DecisionLedgerError(f"duplicate fact_id: {fact.fact_id}")
        self._append_jsonl(self.facts_path, fact.model_dump(mode="json"))

    def append_facts(self, facts: Iterable[DeterministicFact]) -> None:
        for f in facts:
            self.append_fact(f)

    def append_decision(self, decision: ScientificDecisionRecord) -> None:
        """Append a decision. Fails closed if:
          * ``decision_id`` already exists
          * any referenced ``deterministic_facts`` id is unknown
          * ``provenance_class`` is LEGACY_REUSED or TOOL_DEFAULT and
            ``rationale`` is empty or whitespace-only
        """
        if decision.decision_id in self.known_decision_ids():
            raise DecisionLedgerError(
                f"duplicate decision_id: {decision.decision_id}"
            )
        unknown = [fid for fid in decision.deterministic_facts
                   if fid not in self.known_fact_ids()]
        if unknown:
            raise DecisionLedgerError(
                f"decision {decision.decision_id} cites unknown fact_ids: "
                f"{unknown}"
            )
        if decision.provenance_class in {
            ProvenanceClass.LEGACY_REUSED, ProvenanceClass.TOOL_DEFAULT
        } and not (decision.rationale and decision.rationale.strip()):
            raise DecisionLedgerError(
                f"decision {decision.decision_id} has provenance_class="
                f"{decision.provenance_class.value} but empty rationale; "
                f"legacy/tool-default reuse requires explicit justification"
            )
        self._append_jsonl(self.decisions_path, decision.model_dump(mode="json"))

    def append_contradiction(self, contradiction: JudgeContradiction) -> None:
        self._append_jsonl(self.contradictions_path,
                           contradiction.model_dump(mode="json"))

    # ---- audit queries -------------------------------------------------
    def why(self, decision_id: str) -> Optional[ScientificDecisionRecord]:
        """Locate a decision by id (linear scan of the JSONL). Returns
        ``None`` if not found. This is the primitive that answers
        auditor questions like "why this parent count?"."""
        for d in self.iter_decisions():
            if d.decision_id == decision_id:
                return d
        return None

    def decisions_for_stage(self, stage: str) -> list[ScientificDecisionRecord]:
        return [d for d in self.iter_decisions() if d.stage == stage]


__all__ = [
    "DecisionLedger",
    "DecisionLedgerError",
    "DECISIONS_FILENAME",
    "FACTS_FILENAME",
    "CONTRADICTIONS_FILENAME",
]
