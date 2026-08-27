# V2 Manual Verification Packet

This packet lists the commands a human runs to verify the property-guided
distillation V2 hardening before authorizing a real Fresh campaign. Nothing
here runs Teacher inference, Student training, MD, DFT, replay, or benchmark
sweeps; every check is deterministic and offline.

A pre-existing, unrelated baseline failure exists and is **not** a V2
regression:

- `tests/test_v2_regions_sampling_tracking.py::test_old_domain_representation_adapts_to_structural_regions`
  (historical `regions_from_domain_representation` compatibility path returns
  `None` for a frame; present on clean HEAD before this hardening work).

A separate pre-existing collection error is unrelated to V2:

- `tests/test_fe042_coverage_adequacy_gate_control.py` imports a missing module
  (`tests.test_full_lifecycle_integration`). Ignore it with
  `--ignore=tests/test_fe042_coverage_adequacy_gate_control.py`.

## CONFIRMED_EXISTING

Pre-existing suites that must continue to pass (regression guard):

```
pytest tests/test_v2_property_targets.py
pytest tests/test_v2_regions_sampling_tracking.py tests/test_v2_region_evaluation.py
pytest tests/test_stage9_calibrated_uncertainty.py
pytest tests/test_stage10_deployment_plan.py
pytest tests/test_stage11_teacher_physical_validation.py
pytest tests/test_stage12_final_analysis.py
pytest tests/test_architecture_freeze.py
pytest tests/test_pydantic_ai_hpc_verification.py
pytest --ignore=tests/test_fe042_coverage_adequacy_gate_control.py
```

## PROPOSED_NEW

V2 hardening suites added by this work (H01-H08):

```
pytest tests/test_v2_representation_soap.py
pytest tests/test_v2_sampling_and_stopping.py
pytest tests/test_v2_target_operationalization.py
pytest tests/test_v2_region_evaluation_tracking.py
pytest tests/test_v2_recovery_execution_graph.py
pytest tests/test_v2_protected_eligibility.py
pytest tests/test_v2_replay_supercell_bridges.py
pytest tests/test_v2_judge_routing_efficiency.py
pytest tests/test_v2_synthetic_e2e.py
pytest tests/test_v2_region_provider_equivalence.py
pytest tests/test_v2_workflow_integration.py
pytest tests/test_v2_efficiency_evidence.py
```

The `test_v2_synthetic_e2e.py` mock control loop additionally traverses the
paper-facing `V2WorkflowPlan` transitions
(SPECIFY -> DISCOVER -> CURATE -> DISTILL -> TRACK -> RECOVER -> DISTILL ->
TRACK -> VALIDATE -> COMPLETE) using only the H10 transition helpers; it is
still mock-only and executes no backend.

## NEEDS_SOURCE_CONFIRMATION

Runtime-integration suites whose green status confirms the V2 contracts still
interoperate with the existing executors; confirm these before a Fresh run:

```
pytest tests/test_pydantic_ai_executors.py
pytest tests/test_acquisition_lifecycle.py
pytest tests/test_replay_runtime_reproducibility.py
pytest tests/test_fe054_train_augmentation.py
```

## Invariants each layer verifies

- **Eligibility** (`test_v2_protected_eligibility`, `test_v2_replay_supercell_bridges`):
  protected/test data never becomes training, recovery, replay, or augmentation
  eligible; the check fails closed on hash drift or missing split role.
- **Sampling** (`test_v2_sampling_and_stopping`): DIRECT-like is structural
  stratification; under-budget is an explicit unresolved state; selection is
  invariant under region rename.
- **Closure** (`test_v2_sampling_and_stopping`, `test_v2_region_evaluation_tracking`):
  unbound required criterion -> human input; bound+missing -> evidence not
  evaluated; a required target-property failure forces RECOVER even when
  energy/force pass.
- **Evaluation binding** (`test_v2_region_evaluation_tracking`): metrics group
  only by the protected-population binding; training-only members are ignored.
- **Recovery** (`test_v2_recovery_execution_graph`): staged lifecycle with no
  premature future artifacts; Teacher frozen; no new DFT.
- **Efficiency** (`test_v2_region_evaluation_tracking`, `test_v2_judge_routing_efficiency`):
  unknown != zero; measured values carry provenance; no scalar total cost.
- **Cohesion** (`test_v2_synthetic_e2e`, `test_v2_region_provider_equivalence`):
  a mock-only control loop closes a deficient region; explicit/discovered/hybrid
  region providers share one downstream pipeline.
- **Workflow integration** (`test_v2_workflow_integration`,
  `test_v2_efficiency_evidence`): the non-executing `V2WorkflowPlan` cannot
  advance past SPECIFY while operationalization is pending or past CURATE under
  an insufficient selection budget; DISTILL only emits external requests; only
  evaluated `RECOVER` regions route to recovery; final validation requires every
  latest required region CLOSED (not merely `deficient_regions()==[]`); coverage
  values stay `None` (not zero) when unmeasured; Pareto rows keep raw dimensions
  with no scalar total cost; FE-067/FE-068 surfaces are reused via the adapter
  map; the final evidence record binds the frozen-Teacher / no-new-DFT
  invariants and protected-population identity.

Runtime verification with real backends is still required before any scientific
claim is made from a V2 campaign.
