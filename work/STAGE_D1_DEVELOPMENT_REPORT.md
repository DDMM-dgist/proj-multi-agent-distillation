# Stage D-1 development-set replay — result (documentation)

`LOCAL_STAGE_D1_DEVELOPMENT = AUTHORITATIVE_DETERMINISTIC_REPLAY_PASS`

Development-set replay of the 7 auditable frozen scientific decisions, re-run once after the
deterministic-authority hardening. Model `local-openai/qwen2.5-7b-instruct` (vLLM 0.26.0, BF16).
Evaluated OFFLINE with the committed `work/stage_d1_evaluate.py` on the newest provenance per
checkpoint (older attempts in the archive are the pre-hardening FAIL run; selection is by
`recorded_at`). Every newest attempt's `prompt_sha256` equals the SHA-256 of `agents/judge.md` at
the evaluated revision, confirming the run used the hardened Judge spec.

## Result

| checkpoint | det. severity | Judge verdict | agreed w/ det. truth | canonical validation | historical | comparison |
|---|---|---|---|---|---|---|
| d1-committee-v3 | REVISE | REVISE | yes | accepted | REVISE | AGREE |
| d1-committee-v5 | PASS | PASS | yes | accepted | PASS | AGREE |
| d1-data-provenance | REVISE | REVISE | yes | accepted | REVISE | AGREE |
| d1-dft-cc001 | FAIL | FAIL | yes | accepted | FAIL | AGREE |
| d1-dft-cell_001 | PASS | PASS | yes | accepted | PASS | AGREE |
| d1-dft-clustered_cell_002 | PASS | PASS | yes | accepted | PASS | AGREE |
| d1-physical-validation | PASS | PASS | yes | accepted | PASS | AGREE |

- 7/7 typed output; 7/7 canonical deterministic consistency; 7/7 historical agreement.
- UNJUSTIFIED_DIFFERENCE = 0; false_scientific_pass = 0; fabricated_evidence = 0;
  nonexistent_artifact = 0; unauthorized_execution = 0; controller_mutation = 0; paid_api = 0;
  missing_criterion = 0; provenance_complete = 7/7; model_consistency ok.

The two prior failures are resolved by the architecture, not by prompt trust: v5 → PASS
(`0.339 <= 0.376 => True` computed deterministically) and cc001 → FAIL (invalidating physical
predicates false; SCF convergence does not override physical invalidity). Contradictory votes on the
real v5/cc001 tasks are rejected by the committed canonical validator (both boolean reversal and
verdict-vs-severity).

This file is documentation of the development evaluation only. No runtime or semantic code changed to
produce it. The architecture is frozen for holdout evaluation at this revision (see the freeze
manifest recorded in the accompanying commit message / `work/stage_d1_holdout/FREEZE.md`).
