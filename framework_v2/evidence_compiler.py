"""Framework V2 -- stage-generic evidence compiler (Section 14).

R31 grew two bespoke, stage-specific evidence hooks:

  * ``runtimes/pydantic_ai/bounded_evidence.py`` -- bounded, semantic
    summaries of large artifacts (kept and generalized).
  * ``runtimes/pydantic_ai/training_evidence.py`` -- a compact training
    summary whose deterministic ``verification_outcomes`` were hand-authored
    for the training gate, with SiO2/committee-of-4 specifics baked in.

The V2 lesson: the "compile deterministic facts about this stage's artifacts,
bound in size, into the packet the Judges actually see" capability is
stage-generic and must not be re-implemented per stage with per-campaign
constants. This module is that generic compiler.

Layering: ``framework_v2`` is the foundation and must not import the runtime.
So the compiler takes an *injected* artifact summarizer -- the runtime passes
``runtimes.pydantic_ai.bounded_evidence.summarize_artifact`` (the kept,
generalized summarizer); tests and standalone callers can rely on the
built-in minimal summarizer. The authoritative content is a list of
``DeterministicFact`` records (Section 13): the compiler serializes them both
as ``deterministic_facts`` (typed) and as ``validation_outcomes`` (the
``{"check", "ok", ...}`` shape the existing Judge packet already consumes), so
wiring the runtime to this compiler is a drop-in for the old per-stage hooks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from framework_v2.facts import DeterministicFact, FactVerdict

# Default cap mirrors the runtime's bounded-evidence budget (256 KB). The
# compiler fails closed rather than silently truncating structured facts.
DEFAULT_MAX_EVIDENCE_BYTES = 256 * 1024

ArtifactSummarizer = Callable[[str], dict]


def _default_summarizer(path: str) -> dict:
    """Minimal, dependency-free artifact summary (path + size + sha).

    Enough for standalone/tests; the runtime injects the richer semantic
    summarizer. Directories are summarized by aggregate size only (no
    recursive hashing here -- that is the runtime summarizer's job)."""
    p = Path(path)
    summary = {"artifact_path": str(p), "summary_kind": "generic", "evidence_gaps": []}
    if not p.exists():
        summary["evidence_gaps"].append("artifact_missing")
        summary["integrity"] = {"exists": False}
        return summary
    if p.is_dir():
        total = 0
        n = 0
        for child in p.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
                n += 1
        summary["integrity"] = {"kind": "directory", "size": total, "n_files": n}
        return summary
    data = p.read_bytes()
    summary["integrity"] = {
        "kind": "file",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return summary


def fact_to_validation_outcome(fact: DeterministicFact) -> dict:
    """Render a DeterministicFact as a legacy ``validation_outcome`` dict.

    Keeps the ``{"check", "observed", "ok"}`` keys existing Judge-packet
    consumers already read, and adds the typed V2 fields (``fact_id``,
    ``verdict``, ``expected``, ``validator``) so nothing is lost."""
    return {
        "check": fact.kind,
        "fact_id": fact.fact_id,
        "observed": fact.observed,
        "expected": fact.expected,
        "verdict": fact.verdict.value,
        "ok": fact.verdict == FactVerdict.PASS,
        "artifact_sha256": fact.artifact_sha256,
        "validator": fact.validator,
        "rationale": fact.rationale,
    }


class EvidenceCompiler:
    """Assemble one stage's bounded, fact-carrying evidence bundle."""

    def __init__(
        self,
        *,
        summarizer: Optional[ArtifactSummarizer] = None,
        max_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
    ):
        self._summarize = summarizer or _default_summarizer
        self.max_evidence_bytes = int(max_evidence_bytes)

    def compile_stage_evidence(
        self,
        *,
        stage: str,
        artifacts: Iterable[str | Path] = (),
        facts: Sequence[DeterministicFact] = (),
        protocol_refs: Iterable[str | Path] = (),
        extra_validation_outcomes: Iterable[dict] = (),
        scope_contract_sha256: Optional[str] = None,
        out_path: Optional[str | Path] = None,
    ) -> dict:
        """Build the bundle. Deterministic facts are authoritative and are
        rendered both typed (``deterministic_facts``) and legacy-shaped
        (``validation_outcomes``). Fails closed if the serialized bundle
        exceeds ``max_evidence_bytes`` rather than dropping facts."""
        artifact_summaries = [self._summarize(str(a)) for a in artifacts]
        protocol_records = []
        for ref in protocol_refs:
            p = Path(ref)
            if p.exists():
                protocol_records.append({
                    "path": str(p.resolve()),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.is_file() else None,
                })
        fact_list = list(facts)
        validation_outcomes = (
            [fact_to_validation_outcome(f) for f in fact_list]
            + list(extra_validation_outcomes)
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "profile": "framework_v2_stage_evidence",
            "stage": stage,
            "scope_contract_sha256": scope_contract_sha256,
            "max_evidence_bytes": self.max_evidence_bytes,
            "artifacts": artifact_summaries,
            "protocol_refs": protocol_records,
            "deterministic_facts": [f.model_dump(mode="json") for f in fact_list],
            "validation_outcomes": validation_outcomes,
            "all_facts_pass": all(
                f.verdict == FactVerdict.PASS for f in fact_list
            ) if fact_list else True,
            "n_facts": len(fact_list),
            "n_facts_failing": sum(
                1 for f in fact_list if f.verdict == FactVerdict.FAIL),
            "n_facts_unchecked": sum(
                1 for f in fact_list if f.verdict == FactVerdict.UNCHECKED),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if len(text.encode("utf-8")) > self.max_evidence_bytes:
            raise ValueError(
                f"compiled stage evidence for {stage!r} exceeds "
                f"{self.max_evidence_bytes} bytes"
            )
        if out_path is not None:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            payload["summary_path"] = str(out.resolve())
            payload["summary_sha256"] = hashlib.sha256(
                (text + "\n").encode("utf-8")).hexdigest()
        return payload


__all__ = [
    "DEFAULT_MAX_EVIDENCE_BYTES",
    "ArtifactSummarizer",
    "EvidenceCompiler",
    "fact_to_validation_outcome",
]
