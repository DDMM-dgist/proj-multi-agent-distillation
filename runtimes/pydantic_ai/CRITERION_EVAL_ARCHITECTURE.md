# Deterministic criterion evaluation layer (Stage D-1 architecture)

## Why this exists — the two Stage D-1 failures

`LOCAL_STAGE_D1 = AUDITABLE_FROZEN_DECISION_REPLAY_FAIL` (safety gates all zero; historical agreement
5/7). The two disagreements were **architectural**, not model-quality:

1. **`d1-committee-v5` — a deterministic arithmetic error by the LLM.** The Judge asserted
   `0.339 > 0.376` (false) and thereby mis-voted a criterion. A numeric comparison must never be left
   to free-form LLM arithmetic.
2. **`d1-dft-cc001` — a severity/policy-mapping error.** cc001 is an unphysical atom-overlap artifact
   (E/atom +17.29 eV, Fmax 6750, min-distance 0.18 Å) whose historical verdict is **FAIL**. The Judge
   reached a REVISE-class outcome — right that the artifact is bad, wrong on the *severity*, because
   "invalid/unphysical" must map to FAIL, not REVISE.

Neither is fixed by prompt wording or a bigger model. Both are removed by moving the settled parts of
the decision — the arithmetic and the severity policy — **out of the LLM and into deterministic code**.

## Principle

> Raw frozen evidence → **deterministic criterion evaluator** → typed criterion results
> `{criterion, operator, lhs, rhs, result, invalidating, provenance}` → **LLM Judge** (interpretation,
> rationale, verdict) — and the LLM may **not reverse** a deterministic boolean.

Deterministic Python owns: numeric comparisons, thresholds, ranges, missing-field checks, boolean
conjunction/disjunction, and the FAIL/REVISE/PASS **severity policy**. The LLM owns: reading the
evidence, scientific interpretation, and writing the verdict + rationale where those are genuinely
semantic.

**Generic, never an answer key.** The evaluator and every spec operate from structured predicate
definitions over evidence field names (`field_a < field_b`, `field_c <= threshold`,
`required_field exists`). There is **no** `if task_id == "d1-committee-v5": PASS`. A regression test
(`test_no_task_id_special_casing`) asserts the module source contains no `task_id`/`d1-` literal and
every committed spec uses only generic predicate keys.

## Components

| File | Role |
|---|---|
| `runtimes/pydantic_ai/criterion_eval.py` | The layer: `CriterionResult`, `evaluate_criterion`, `evaluate_criteria`, `derive_severity`, `render_authoritative_block`, `attach_to_task`. |
| `work/stage_d1_gen_criteria.py` | Generates the **generic** criterion specs (predicate families only) per checkpoint, aligned 1:1 with each task's ordered criteria. |
| `work/stage_d1_attach_criteria.py` | Integration step: evaluates evidence×specs and injects the typed results into each Judge task's `context` (upstream of the LLM). |
| `agents/judge.md` | Contract: when `deterministic_criterion_results` is present, the booleans are authoritative — `criteria_checked.ok` must match, never recompute/reverse, verdict from the severity policy. |
| `tests/test_pydantic_ai_criterion_eval.py` | 10 regression tests (below). |

### Criterion spec (input, generic)

```json
{ "criterion": "<human text>", "operator": "le",
  "lhs": {"field": "error_c_eV_A"}, "rhs": {"const": 0.368}, "invalidating": false }
```
Operators: `le lt ge gt eq ne exists not_exists in_range approx`. Operands: `{"field": name}` |
`{"const": value}` | bare literal. `in_range` → `rhs {low,high}`; `approx` → `rhs {value,tol}`.
Compound: `{"all": [...]}` / `{"any": [...]}`. `invalidating: true` marks a physical-validity
predicate whose failure is disqualifying (→ FAIL).

### CriterionResult (typed output)

```
criterion:str  operator:str  lhs:Any  rhs:Any  result:bool  invalidating:bool  provenance:str
```
`provenance` shows the settled computation, e.g. `"0.339 <= 0.376 => True"`,
`"-11 <= 17.29 <= -8 => False"`, `"MISSING_FIELD:x => False"`.

### Severity policy (`derive_severity`) — general, source-grounded

- a **failed invalidating** criterion ⇒ **FAIL** (invalid/unphysical blocks);
- else all criteria true ⇒ **PASS**;
- else (a non-invalidating criterion unmet/unverifiable) ⇒ **REVISE** (salvageable).

Grounding: `gates/README.md` ("any judge votes FAIL → FAIL (invalid/unphysical artifact blocks
regardless of rule)"; PASS = all met; otherwise REVISE) and `agents/judge.md` (FAIL = invalid/unphysical;
REVISE = salvageable, a criterion unmet/unverifiable). In the historical trail the **only** FAIL is
cc001 (unphysical), and every REVISE is incomplete/regression — so cc001 → FAIL is *derivable from the
general policy*. cc001 therefore **remains an UNJUSTIFIED_DIFFERENCE** (a severity error the layer
fixes), and the frozen expectation/evaluator is **not** changed.

### Fail-closed behaviour

A missing operand never crashes and never silently passes: the result is `False` with a
`MISSING_FIELD:` provenance. Type mismatch → `False` with `TYPE_MISMATCH:`. Unknown operator →
`False` with `BAD_OPERATOR:`.

## Integration (upstream of the Judge)

`attach_to_task(task, results, *, authoritative=True)` writes into `task["context"]`:
`deterministic_criterion_results` (list of `CriterionResult`), `deterministic_suggested_severity`,
`deterministic_authoritative` (the gate MODE, see below), and a `deterministic_note`. The runtime
already serializes the full task (context included) into the model input (`pydantic_ai_runtime.run` →
`agent.run_sync(json.dumps({"task": task}))`), so the Judge receives the authoritative facts before it
reasons. `judge.md` instructs the Judge to obey them. This is the whole fix for failure (1): the model
can no longer be the arbiter of `0.339 <= 0.376`.

## Structural enforcement (not just prompt instruction)

Prompt instruction is necessary but not sufficient — a model can still emit a contradicting vote. The
**canonical post-model validator** rejects it, so the guarantee holds structurally on the acceptance
path (Pydantic parse → `validate_agent_response` → `validate_judge_vote`, the same path FileExchange
and the driver use). In `orchestration/exchange.py`:

- `validate_agent_response` extracts the block from `task.context` and passes it to
  `validate_judge_vote(..., deterministic=...)`.
- `_enforce_deterministic(payload, checked, criteria, deterministic)` then rejects a JudgeVote that:
  - reverses a computed boolean in **either** direction (`criteria_checked[i].ok != result[i]`) —
    including a missing-value result (`MISSING_FIELD → False`), so a missing value can never be
    converted into an unsupported positive;
  - does not cover every ordered criterion with a deterministic result, or whose block criterion
    identity/order does not match the task criteria (a dropped or extra criterion is a mismatch — this
    composes with the pre-existing `checked_names == criteria` check);
  - for a **fully deterministic** gate, carries a verdict `!= deterministic_suggested_severity`.

This is general: `_enforce_deterministic` keys off criterion order + identity + the mode flag, never a
task id (regression-tested). It is the whole fix for failure (2): `cc001` (deterministic severity FAIL)
cannot be accepted as REVISE/PASS.

### Gate MODE (explicit in the typed context)

- `deterministic_authoritative: true` — **fully deterministic** gate (every criterion is a
  numeric/physical/boolean predicate; all Stage D-1 gates). Deterministic truth is binding: both the
  criterion booleans and the verdict are enforced. Deterministic truth is **not left advisory**.
- `deterministic_authoritative: false` — **advisory** block for a gate with genuinely semantic
  criteria: provided for reference, not enforced; the Judge supplies the semantic verdict. A mixed gate
  should be split into deterministic (authoritative) + semantic criteria rather than mislabel a numeric
  criterion as advisory.

## Verification (deterministic, network-free)

Regression tests in `tests/test_pydantic_ai_criterion_eval.py`:
- `0.339 <= 0.376 => True` (the required case) and a reversed comparison **cannot occur** (the boolean
  is computed; the strictly-wrong `gt` is deterministically `False`).
- missing value handled deterministically (→ `False`, `MISSING_FIELD`; `exists`/`not_exists` correct).
- compound `all`/`any` handled correctly.
- `in_range` / `approx`.
- general severity policy (invalidating-fail→FAIL, all-true→PASS, else→REVISE).
- the committed D1 specs on D1 evidence reproduce **every** historical severity generically — critically
  v5 → PASS (`0.339 <= 0.376 => True`) and cc001 → FAIL (invalidating physical predicates).
- `attach_to_task` injects the authoritative context without mutating the original; the committed D1
  tasks carry a block whose booleans + severity equal a fresh evaluation and the historical severity.
- `judge.md` states the results are authoritative and must not be recomputed/reversed.
- **no task-ID special-casing** in the module or any spec.

Enforcement regression tests in `tests/test_authoritative_criterion_enforcement.py` drive the canonical
`validate_agent_response` path: deterministic-true + vote-false → rejected; deterministic-false +
vote-true → rejected; deterministic FAIL + vote REVISE/PASS → rejected; deterministic PASS + vote
REVISE/FAIL → rejected; a fully consistent vote → accepted; missing value cannot be flipped positive;
missing/extra criterion → rejected; advisory block is not verdict-binding; no task-ID special-casing.

Deterministic severity now MATCHES historical for all 7 development checkpoints (was 5/7 with the LLM
doing the arithmetic). This does **not** by itself certify generalization — see below.

## v2 refactor — deterministic-verdict OWNERSHIP (post-holdout)

The Stage D-1 holdout replay returned an **LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE**: on
`hd-committee-v3final` the Judge read the evidence correctly and got all criterion booleans right, but
emitted **FAIL** where the deterministic policy gives **REVISE**; the v1 enforcement *rejected* the
vote, so canonical consistency was 7/8. The failure was not evidence-grounding, arithmetic, safety, or
policy — it was that the architecture asked the LLM to **regenerate** a verdict the deterministic
policy already owns, and validated the copy. That redundant step is an avoidable liveness/consistency
risk.

**Fix — move the ownership boundary.** For a **fully deterministic** gate
(`deterministic_authoritative = true`):

1. the deterministic **criterion evaluator** owns the criterion booleans;
2. the deterministic **policy** (`derive_severity`) owns the **authoritative final verdict**;
3. the **LLM does not own or override** the verdict;
4. the LLM produces only **interpretation** (rationale, per-criterion commentary, concerns, remediation);
5. the accepted scientific verdict is **bound from the deterministic policy by trusted code**.

**Typed separation** (`criterion_eval.py`): `DeterministicGateDecision{criterion_results,
authoritative_verdict, provenance}` (produced by trusted code) vs `JudgeInterpretation{rationale,
criterion_commentary, concerns, recommended_remediation}` (the LLM's contribution).

**Binding** (`orchestration/exchange.py::bind_authoritative_judge_vote`, applied inside
`validate_agent_response` and recorded by `production_router`): for an authoritative gate the accepted
JudgeVote's `verdict` is set to `deterministic_suggested_severity` and its `criteria_checked` is rebuilt
from the ordered criteria + the deterministic booleans; the LLM's `rationale`/`required_fix` are kept;
its proposed verdict and any criterion-fact contradiction are recorded in provenance
(`llm_proposed_verdict`, `verdict_overridden`, `criterion_contradictions`, `accepted_verdict`) but are
**not** authoritative. A verdict-wording difference no longer fails the gate; a contradictory criterion
claim is **overridden and flagged**, never accepted. **Advisory** gates
(`deterministic_authoritative = false`) keep the genuine semantic Judge verdict path unchanged.

Safety is preserved, not weakened: the accepted verdict + booleans are now *always* the deterministic
ones (the LLM cannot set or contradict them), FAIL/REVISE/PASS semantics are unchanged, and
canonical validation, request_limit=6, the duplicate-read guard, provenance, authorization, and
controller authority all remain. Regression tests: authoritative REVISE/FAIL/PASS cannot be changed by
the LLM; a final verdict is always produced even if the LLM's wording/structure differs; contradictory
commentary is overridden + flagged; advisory gates still take a genuine LLM verdict; and a 15-case
network-free corpus (7 dev + 8 consumed-holdout) binds every case to the deterministic verdict against
an adversarial LLM vote. The evaluator reads `accepted_verdict` (falling back to the raw verdict for
pre-refactor archives, so historical results are unchanged).

## Holdout requirement (why 7/7 here is necessary, not sufficient)

The 7 checkpoints are now a **development set** (the layer + specs were observed against them). Final
acceptance additionally requires an **untouched holdout** replay — inventory in
`work/stage_d1_holdout_inventory.md`, verdicts sealed, not built or run yet. Acceptance =
development replay pass **AND** holdout replay pass **AND** all hard safety gates zero **AND** no
UNJUSTIFIED_DIFFERENCE. No live inference is run by this change.
