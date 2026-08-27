# Property-Guided Distillation V2

V2 presents the paper-facing workflow as:

```text
Human-specified target physics
        |
        v
Target operationalization
        |
        v
Structural region discovery
        |
        v
Minimal initial curation
        |
        v
Teacher labeling
        |
        v
Student committee distillation
        |
        v
Protected evaluation
        |
        v
Region-resolved ErrorTracker
        |
        +---- deficient? ---- YES ----+
        |                             |
       NO                     Targeted curation
        |                             |
        v                             |
Final target validation <---- Re-distillation
        |
        v
Deployment / final evidence
```

The human specifies the target physics; the agent operationalizes it. The
agent may select observables from the allowed registry, retrieve methodology,
and bind evidence-backed criteria. It must not replace the target family,
silently broaden the objective, or invent thresholds after seeing results.

Validation is actionable feedback, not merely endpoint reporting. Protected
evaluation identifies weak configuration-space regions; recovery selects
similar eligible training-side structures. Protected frames are never recycled
into training, DFT replay, augmentation parents, or threshold fitting.

The Teacher remains frozen during Student recovery. V2 excludes Teacher
retraining, new-DFT active learning, and post-deployment Teacher refinement.

Candidate generation and candidate selection are separate. Generation may come
from a Teacher DB, perturbation, augmentation, supercells, future MD, or future
AGOX/global-search backends. Selection may use Random, FPS, DIRECT-like
stratification, uncertainty, or uncertainty plus diversity.

Initial curation and recovery curation use different available information.
Initial curation has no trained Student and therefore uses representation,
region coverage, and diversity. Recovery curation may additionally use
region-resolved Student error and committee disagreement.

The old 12-stage engine remains available for historical campaigns and
internal safety. It is not the V2 paper-facing scientific contribution.

## Public Concepts

- Specify: `HumanTargetPropertyContract`, `TargetOperationalization`
- Discover: `StructuralRepresentation`, `StructuralRegionProvider`
- Curate: candidate generation and sampler interfaces
- Distill: Teacher labeling and Student committee training
- Track: protected evaluation, `ErrorLedger`, raw efficiency records
- Recover: `RegionRecoveryPlan`, targeted curation
- Validate: target-property validation, deployment validation, final evidence

## Implementation Map

- Human target and observable registry:
  `framework_v2/property_targets.py`
- Structural region public abstraction:
  `framework_v2/structural_regions.py`
- Representation and discovery:
  `framework_v2/structural_representation.py`,
  `framework_v2/region_discovery.py`
- Samplers and stopping policy:
  `framework_v2/v2_sampling.py`
- Error and efficiency ledgers:
  `framework_v2/error_tracking.py`
- Region recovery:
  `framework_v2/region_recovery.py`
- Replay and supercell experiment controls:
  `framework_v2/experiment_controls.py`
- Deterministic-first Judge policy:
  `framework_v2/v2_judge_policy.py`

## Maturity: Scaffold vs Implementation vs Runtime vs Verification

These four layers are distinct and must not be conflated:

- **Scaffold** — the paper-facing workflow shape and public contract surface
  (the modules in the Implementation Map). Present and hardened.
- **Implementation** — the deterministic Python contracts and pure functions
  that encode the invariants (eligibility, closure, staged recovery,
  efficiency accounting). Present and unit-tested.
- **Runtime integration** — wiring these contracts to real executors
  (Teacher inference, Student committee training, MD, DFT). Adapters exist;
  a real Fresh run has **not** been executed and requires human operational
  input plus HPC approval.
- **Verification** — a real end-to-end campaign producing scientific evidence.
  Only synthetic, mock-only control loops have run so far
  (`tests/test_v2_synthetic_e2e.py`). Runtime verification is still required
  before any scientific claim.

## Sampler Semantics

- **DIRECT-like** is *structural-stratified diversity selection*: per-region
  coverage first, residual filled by global FPS when a representation is
  present or a stable round-robin quota when it is absent. It is **not** the
  published DIRECT method and must not be cited as such.
- **Under-budget is an explicit unresolved state, not a silent truncation.**
  When the requested count cannot be met from eligible candidates the sampler
  returns `SelectionStatus.SELECTION_BUDGET_INSUFFICIENT` with an empty
  selection and an `unresolved_reason`, rather than returning an arbitrary
  alphabetical subset.
- Selection is invariant under region-ID renaming: relabeling regions does not
  change which candidates are chosen.

## Structural Representation (SOAP dependency behavior)

The SOAP backend depends on the optional `dscribe` package. Representation is
content-hashed and stable across runs. If a requested backend's dependency is
unavailable the representation adapter fails closed with a typed error rather
than silently degrading to a different descriptor; the composition backend has
no third-party dependency and is always available.

## Protected Evaluation Binding

`EvaluationPopulationRegionBinding` binds the frozen, hash-pinned protected
evaluation population to structural regions. Metric aggregation groups **only**
by this binding's `frame_to_region` map, so training-only members never leak
into evaluation metrics. A required region with no evidence, a missing required
frame, or mixed reference channels fails closed. Training eligibility is derived
by the single authoritative invariant in `framework_v2/protected_eligibility.py`:
a candidate is training-eligible only if it is not protected and carries the
TRAIN split role in the frozen split; protected/test data can never enter
training, recovery candidates, replay, or augmentation parents.

## Staged Recovery Lifecycle

Recovery is an ordered, state-guarded execution graph, not a single leap:

```text
PLANNED -> LABELS_READY -> DATASET_READY -> STUDENT_READY -> EVALUATION_READY
```

Each transition authorizes exactly one next request (Teacher labeling ->
training-dataset update -> redistillation -> next evaluation). A downstream
artifact may exist only once its producing transition has run — the bundle
validator rejects any bundle that pretends a future artifact already exists.
Redistillation keeps the Teacher frozen and requests no new DFT.

## Efficiency: Zero vs Unknown

Raw efficiency dimensions are kept separate; there is no arbitrary scalar total
cost. Every numeric field is optional and defaults to `None` meaning
**unknown**, which is distinct from a measured `0`. A measured value must carry
`measurement_provenance`; cost evidence is therefore never fabricated. Judge
routing surfaces its own accounted cost (zero Judge/LLM calls for a
deterministic gate, exactly one for an allowed scientific-ambiguity reason).

## Historical Stage-8 Identities Preserved

The old 12-stage engine and its Stage-8 population/domain identities remain
byte-stable for historical campaigns and internal safety. V2 adapts that
historical evidence to provider-neutral regions without renaming, re-scoring,
or promoting historical diagnostic frames into primary pass/fail logic.
