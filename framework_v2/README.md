# Framework V2

Framework V2 is the scientific-decision architecture that lives beneath the
existing 12-stage distillation Controller. The Controller keeps orchestrating
lifecycle (state, gates, recovery, adjudication). What V2 adds is a set of
first-class, typed, artifact-addressable **scientific contracts** and the
**generic capabilities** that consume them, so R31-class failures cannot silently
recur.

**Status:** feature-complete for freeze. The foundational contracts, decision
ledger, deterministic-fact/judgment distinction, convergence policy, capability
negotiation, scope-aware evaluation, blind-test enforcement, student-recipe
provenance validator, generic domain discovery, representative + leakage-safe
partition validator, the late-lifecycle policy validators (K/L/M), and the
stage-generic evidence compiler are all implemented and covered by the
`tests/framework_v2_regression/` suite. The V2 modules are wired into the real
Controller gate path (`record_gate`) and the real PydanticAI dispatch path
(`authorize_and_execute`); `tests/framework_v2_regression/test_controller_integration.py`
exercises those live paths end to end.

## Design principles

1. **Every scientific choice has a schema.** No scientific decision is
   carried only in prose prompts, YAML keys, or executor internals.
2. **Every contract has a canonical SHA.** Downstream contracts bind to
   upstream ones by content-hash. Silent scope drift is detectable.
3. **Deterministic facts are authoritative.** An LLM Judge cannot negate
   a fact produced by a validator against real artifact content. Doing
   so produces a `JUDGE_CONTRADICTION`, which is *not* usable as scientific
   REVISE evidence.
4. **Fail closed on capability gaps.** A planner requiring capabilities the
   executor does not advertise produces `FRAMEWORK_CAPABILITY_BLOCKER` --
   the plan is not silently reduced.
5. **Provenance is per-parameter.** Every scientific-critical value carries
   `{value, provenance_class, evidence, alternatives_considered, rationale}`.
   `LEGACY_REUSED` and `TOOL_DEFAULT` are allowed but require explicit
   rationale.

## Sub-modules

| module                     | Directive § | Purpose |
|----------------------------|-------------|---------|
| `contracts.py`             | 2           | 13 typed contracts A..M (all pydantic v2, `.content_sha256()`) |
| `facts.py`                 | 13          | `DeterministicFact` / `ScientificJudgment` / `JudgeContradiction` |
| `decision_ledger.py`       | 16          | Append-only, fail-closed provenance ledger |
| `convergence.py`           | 10          | `ConvergencePolicy` classifier: `NOT_CONVERGED / CONVERGED_AT_MAX / CONVERGED_EARLY / INSUFFICIENT_DATA` |
| `capability.py`            | 17          | `FRAMEWORK_CAPABILITY_BLOCKER` planner/executor negotiator |
| `evaluation.py`            | 11          | Scope-aware evaluation partitioner; mixed aggregates cannot be primary |
| `blind_test.py`            | 4           | Fail-closed `BlindTestBoundary` + `guard_blind_access` + access log |
| `recipe.py`                | 9           | `validate_recipe_provenance`: rejects silent LEGACY_REUSED |
| `domain_discovery.py`      | 5           | Generic evidence-driven regime discovery producing `DomainRepresentation` |
| `partition_validator.py`   | 8           | Representative + leakage-safe split validator (`PASS_SPLIT / REVISE_SPLIT / LINEAGE_LEAKAGE`) |
| `policy_validators.py`     | 12          | Deterministic validators for the K/L/M policy contracts |
| `evidence_compiler.py`     | 14          | Stage-generic compiler generalizing R31 bounded_evidence + training_evidence |

## The 13 contracts (Section 2)

Each has a `schema_version`, immutability (`frozen=True`), `extra="forbid"`,
and `.content_sha256()` (canonical JSON: `sort_keys=True`, compact separators).

* **A. `DeploymentScopeContract`** — objective + list of `ScopeRegion`. The
  single source of truth for scope. Five categories:
  `PRIMARY_DEPLOYMENT`, `AUXILIARY_SUPPORT`, `OUT_OF_SCOPE`,
  `PROTECTED_REFERENCE`, `BLIND_TEST`. At least one primary region required.
* **B. `ScientificDecisionRecord`** — one row of the DecisionLedger.
* **C. `DomainRepresentation`** — `kind ∈ {continuous, categorical,
  hierarchical, hybrid}` + list of `DomainRegime`.
* **D. `CoveragePlan`** — descriptor + distance + stopping criterion.
* **E. `ParentSelectionPlan`** — selector + selected identities.
* **F. `AugmentationPlan`** — **per-parent** `PerParentAugPolicy` list.
  `is_heterogeneous()` drives `augmentation_capability_requirements`.
* **G. `DatasetPartitionPlan`** — lineage key + stratification variables +
  representativeness requirement. Fractions must sum to 1.
* **H. `StudentRecipePlan`** — nine required `RecipeParameter` fields plus
  `additional`. Each parameter carries provenance + evidence + rationale.
* **I. `ConvergencePolicy`** — thresholds for the classifier. NO numbers
  hard-coded; every value carries `provenance_class` +
  `provenance_source`.
* **J. `EvaluationPolicy`** — `primary_metrics`, `diagnostic_metrics`,
  `reject_mixed_aggregate_as_primary` (default `True`).
* **K. `UncertaintyPolicy`**, **L. `DeploymentMDPolicy`**,
  **M. `PhysicalValidationPolicy`** — each has a deterministic validator in
  `policy_validators.py` producing `DeterministicFact` records and a single
  `PASS / REVISE / FAIL` verdict (a missing required input is `REVISE`, an
  evaluated tolerance breach is `FAIL`).

## R31 lessons → V2 safeguards

| R31 failure                                              | V2 safeguard | Regression test |
|----------------------------------------------------------|--------------|-----------------|
| max-epoch treated as converged                           | `ConvergencePolicy` classifier + `convergence_gate_ok` | CASE C |
| Global-count flattening of per-parent aug                | `AugmentationPlan.is_heterogeneous()` + capability negotiator + `FRAMEWORK_CAPABILITY_BLOCKER` | CASE A / L |
| Mixed-scope evaluation aggregate                         | Scope-aware `build_evaluation_report`; validators refuse promoting non-primary partitions to primary | CASE D |
| LLM Judge negating a deterministic fact                  | `detect_judge_contradictions` + `JUDGE_CONTRADICTION` status | CASE E |
| Blind artifact accessed before final eval                | `BlindTestBoundary` + `guard_blind_access` fail-closed + append-only log | CASE F |
| Acquisition scope != evaluation scope                    | `cross_stage_scope_consistent(*shas)` | CASE G |
| Scientific-critical param via silent legacy reuse        | `validate_recipe_provenance` rejects `LEGACY_REUSED`/`TOOL_DEFAULT` without rationale | CASE H |
| Decision cannot be traced back to evidence               | `DecisionLedger.append_decision` fails closed on unknown fact refs | CASE K |
| Bounded semantic evidence for large directories          | *pre-existing* `runtimes/pydantic_ai/bounded_evidence.py` (kept) | pre-existing |
| Recovery hallucinating an artifact path                  | *pre-existing* `runtimes/pydantic_ai/root_cause.py` grounded near-match (kept) | pre-existing |

## Controller + runtime integration

The V2 modules are consumed by the real lifecycle, not just standalone tests:

* **Gate enforcement** — `workflow.controller.RunController.record_gate` calls
  `_enforce_v2_gate_preconditions` before recording any PASS. When V2 is enabled
  (`bind_v2_scope_contract`) and a `convergence_policy` is bound to a stage
  (`bind_v2_contract(..., stage=...)`), the gate builds a convergence report from
  the run's committee LOGs and refuses PASS unless `convergence_gate_ok`. A vote
  bundle carrying structured `v2_judgments` + `v2_facts` is checked for
  `JUDGE_CONTRADICTION`, which also refuses PASS. When V2 is disabled the method
  is a complete no-op, so pre-V2 runs are byte-for-byte unchanged. All Controller
  V2 state lives under an additive `framework_v2` key (schema version unchanged);
  `framework_v2` is imported lazily so the core install (no pydantic) is
  unaffected.
* **Capability negotiation** — `runtimes.pydantic_ai.dispatch.authorize_and_execute`
  runs a capability-negotiation step (5b) before the trusted executor. A proposal
  declaring `required_capabilities` the executor's `supported_capabilities` does
  not advertise is rejected `BLOCKED_CAPABILITY` (via `check_capabilities`); the
  plan is never silently downgraded. No requirement declared → no-op.
* **Evidence compilation** — `runtimes.pydantic_ai.bounded_evidence.build_bounded_evidence`
  accepts `DeterministicFact` records and renders them through
  `evidence_compiler.fact_to_validation_outcome`, emitting both the typed
  `deterministic_facts` block and the legacy `validation_outcomes` shape the Judge
  packet already consumes. The `EvidenceCompiler` class takes an injected
  summarizer so the runtime can supply `bounded_evidence.summarize_artifact`.

See `tests/framework_v2_regression/test_controller_integration.py` for the live
end-to-end coverage of all three integration points.
