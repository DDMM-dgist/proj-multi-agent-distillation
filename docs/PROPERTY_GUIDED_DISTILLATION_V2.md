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
