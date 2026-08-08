"""Deterministic criterion evaluation layer (Stage D-1 architecture; general, not case-specific).

Numeric/boolean scientific predicates are evaluated by DETERMINISTIC Python from structured criterion
definitions — NOT by free-form LLM arithmetic (which produced the Stage D-1 `0.339 > 0.376` error).
The pipeline is:

    raw frozen evidence  ->  evaluate_criteria(evidence, specs)  ->  [CriterionResult ...]  ->  Judge

Each CriterionResult carries lhs / operator / rhs / result(bool) / provenance, and the LLM Judge is
given these as AUTHORITATIVE facts (it interprets + selects the verdict/rationale but must NOT reverse
a deterministic boolean). Severity mapping (FAIL vs REVISE vs PASS) also follows a GENERAL policy that
matches the real gate records + the Judge contract:
  - gates/README.md: "any judge votes FAIL -> FAIL (invalid/unphysical artifact blocks regardless)";
    PASS = all criteria met; otherwise REVISE.
  - agents/judge.md: FAIL = invalid/unphysical (e.g. energy from overlapping atoms); REVISE = salvageable
    (a criterion unmet/unverifiable). => a FAILED "invalidating" (physical-validity) criterion => FAIL.

This module encodes GENERIC predicates + policy. It NEVER encodes a per-task answer (no `if task==...`).
Specs reference evidence fields by name; the module knows nothing about specific checkpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

NonEmptyStr = str

_OPERATORS = {"le", "lt", "ge", "gt", "eq", "ne", "exists", "not_exists", "in_range", "approx"}
_MISSING = object()   # sentinel for an absent evidence field


class CriterionResult(BaseModel):
    """Deterministic evaluation of ONE criterion. The Judge may not reverse ``result``."""
    model_config = {"extra": "forbid"}
    criterion: NonEmptyStr
    operator: str
    lhs: Any = None
    rhs: Any = None
    result: bool
    invalidating: bool = False        # a failed invalidating criterion => severity FAIL
    provenance: str                   # e.g. "0.339 <= 0.376 => True" or "MISSING_FIELD:x => False"


def _resolve(evidence: dict, ref):
    """Resolve an operand reference to a concrete value (or the _MISSING sentinel)."""
    if isinstance(ref, dict):
        if "field" in ref:
            return evidence.get(ref["field"], _MISSING)
        if "const" in ref:
            return ref["const"]
    return ref   # a bare literal


_SYM = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "eq": "==", "ne": "!="}


def evaluate_criterion(evidence: dict, spec: dict) -> CriterionResult:
    """Evaluate one criterion spec against evidence, fully deterministically.

    spec keys: ``criterion`` (text), ``operator`` (see _OPERATORS), ``lhs``/``rhs`` (each a
    {"field": name} | {"const": value} | literal; ``in_range`` uses rhs {"low","high"}; ``approx``
    uses rhs {"value","tol"}), optional ``invalidating`` (bool), and compound ``all``/``any`` (a list
    of sub-specs). Missing operands never crash: a numeric comparison with a missing operand is False
    with a MISSING_FIELD provenance (fail-closed)."""
    text = spec.get("criterion", "")
    invalidating = bool(spec.get("invalidating", False))

    # compound conjunction/disjunction of sub-criteria
    for combiner, reducer in (("all", all), ("any", any)):
        if combiner in spec:
            subs = [evaluate_criterion(evidence, s) for s in spec[combiner]]
            res = reducer(s.result for s in subs)
            prov = f"{combiner}({', '.join(s.provenance for s in subs)}) => {res}"
            return CriterionResult(criterion=text or combiner, operator=combiner, result=bool(res),
                                   invalidating=invalidating, provenance=prov,
                                   lhs=[s.result for s in subs], rhs=None)

    op = spec.get("operator")
    if op not in _OPERATORS:
        return CriterionResult(criterion=text, operator=str(op), result=False,
                               invalidating=invalidating, provenance=f"BAD_OPERATOR:{op} => False")

    lhs = _resolve(evidence, spec.get("lhs"))
    if op == "exists":
        res = lhs is not _MISSING
        return CriterionResult(criterion=text, operator=op, lhs=None if lhs is _MISSING else lhs,
                               result=res, invalidating=invalidating,
                               provenance=f"exists({spec.get('lhs')}) => {res}")
    if op == "not_exists":
        res = lhs is _MISSING
        return CriterionResult(criterion=text, operator=op, lhs=None if lhs is _MISSING else lhs,
                               result=res, invalidating=invalidating,
                               provenance=f"not_exists({spec.get('lhs')}) => {res}")

    if lhs is _MISSING:
        return CriterionResult(criterion=text, operator=op, lhs=None, rhs=spec.get("rhs"),
                               result=False, invalidating=invalidating,
                               provenance=f"MISSING_FIELD:{spec.get('lhs')} => False")

    if op == "in_range":
        rng = spec.get("rhs") or {}
        low, high = rng.get("low"), rng.get("high")
        res = isinstance(lhs, (int, float)) and low <= lhs <= high
        return CriterionResult(criterion=text, operator=op, lhs=lhs, rhs=[low, high], result=bool(res),
                               invalidating=invalidating, provenance=f"{low} <= {lhs} <= {high} => {res}")
    if op == "approx":
        spc = spec.get("rhs") or {}
        val, tol = spc.get("value"), spc.get("tol", 0)
        res = isinstance(lhs, (int, float)) and abs(lhs - val) <= tol
        return CriterionResult(criterion=text, operator=op, lhs=lhs, rhs={"value": val, "tol": tol},
                               result=bool(res), invalidating=invalidating,
                               provenance=f"|{lhs} - {val}| <= {tol} => {res}")

    rhs = _resolve(evidence, spec.get("rhs"))
    if rhs is _MISSING:
        return CriterionResult(criterion=text, operator=op, lhs=lhs, rhs=None, result=False,
                               invalidating=invalidating, provenance=f"MISSING_FIELD:{spec.get('rhs')} => False")
    try:
        cmp = {"le": lhs <= rhs, "lt": lhs < rhs, "ge": lhs >= rhs, "gt": lhs > rhs,
               "eq": lhs == rhs, "ne": lhs != rhs}[op]
    except TypeError:
        return CriterionResult(criterion=text, operator=op, lhs=lhs, rhs=rhs, result=False,
                               invalidating=invalidating, provenance=f"TYPE_MISMATCH:{lhs} {op} {rhs} => False")
    return CriterionResult(criterion=text, operator=op, lhs=lhs, rhs=rhs, result=bool(cmp),
                           invalidating=invalidating, provenance=f"{lhs} {_SYM[op]} {rhs} => {cmp}")


def evaluate_criteria(evidence: dict, specs: list) -> list:
    """Evaluate an ordered list of criterion specs. Order is preserved (mirrors ordered_criteria)."""
    return [evaluate_criterion(evidence, s) for s in specs]


def derive_severity(results: list) -> str:
    """GENERAL severity policy (matches gates/README.md + the Judge contract), not case-specific:
    - a FAILED invalidating (physical-validity) criterion => FAIL (invalid/unphysical blocks);
    - else all criteria met => PASS;
    - else (a non-invalidating criterion unmet/unverifiable) => REVISE (salvageable)."""
    if any((not r.result) and r.invalidating for r in results):
        return "FAIL"
    if results and all(r.result for r in results):
        return "PASS"
    return "REVISE"


def render_authoritative_block(results: list) -> str:
    """A compact, authoritative facts block to inject into the Judge context. The Judge MUST treat
    each boolean as final (it may interpret/verdict but cannot reverse a deterministic result)."""
    lines = ["DETERMINISTIC_CRITERION_RESULTS (authoritative; do NOT recompute or reverse):"]
    for r in results:
        lines.append(f"- [{'PASS' if r.result else 'FAIL'}] {r.criterion} :: {r.provenance}"
                     + (" (invalidating)" if r.invalidating else ""))
    lines.append(f"suggested_severity = {derive_severity(results)}")
    return "\n".join(lines)


_AUTHORITATIVE_NOTE = (
    "These criterion booleans were computed DETERMINISTICALLY from the evidence and are "
    "AUTHORITATIVE: set each criteria_checked.ok to the matching result in order, and NEVER "
    "recompute or reverse a numeric comparison. Decide the verdict from the severity policy "
    "(a failed invalidating result => FAIL; all results true => PASS; otherwise REVISE); your "
    "job is the scientific interpretation and rationale, not the arithmetic.")


def attach_to_task(task: dict, results: list, *, authoritative: bool = True) -> dict:
    """Return a copy of a Judge task with the deterministic criterion results injected into its
    ``context`` as authoritative facts (this is the integration point that puts the deterministic
    layer UPSTREAM of the LLM Judge: the runtime serializes ``task`` — context included — into the
    model input, so the Judge receives the booleans it may not reverse). General: no per-task logic;
    ``results`` is whatever ``evaluate_criteria`` produced for this task's evidence + specs.

    ``authoritative`` records the gate MODE in the typed context so the canonical validator can
    enforce it (see ``orchestration.exchange.validate_judge_vote``):
      - ``True``  — a FULLY DETERMINISTIC gate (all criteria are numeric/physical/boolean predicates,
        as with the Stage D-1 gates): every ``criteria_checked.ok`` must equal the computed result AND
        the verdict must equal ``deterministic_suggested_severity``. Deterministic truth is binding.
      - ``False`` — an ADVISORY block for a gate with genuinely semantic criteria: the block is
        provided for reference but is not verdict-binding; the Judge supplies the semantic verdict.
    Mixed gates should split into deterministic (authoritative) + semantic criteria explicitly rather
    than mislabel a numeric criterion as advisory."""
    out = dict(task)
    ctx = dict(out.get("context") or {})
    ctx["deterministic_criterion_results"] = [r.model_dump() for r in results]
    ctx["deterministic_suggested_severity"] = derive_severity(results)
    ctx["deterministic_authoritative"] = bool(authoritative)
    ctx["deterministic_note"] = _AUTHORITATIVE_NOTE
    out["context"] = ctx
    return out
