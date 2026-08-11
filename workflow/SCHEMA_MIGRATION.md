# Run-manifest schema v6 → v7

## What changed (additive only)
v7 adds operational metadata; **no** stage/gate/retry/recovery scientific semantics change.

Top-level, safe empty defaults:
- `runtime_attempts: []` — references to PydanticAI runtime invocation attempts (the provenance
  itself lives in `exchange/provenance/`); each entry: `task_id, attempt_id, provenance_path,
  role, stage, correlation_id, failure_category, recorded_at`.
- `idempotency: {}` — executed/authorized action idempotency keys → `{action_type, status,
  artifact_ref, recorded_at}` (duplicate-action guard).

Per running stage (only when started via `begin_stage_execution`), a `runner` object:
`{pid, runner_id, started_at, last_update[, interrupted_at]}` for stale-running detection.

## Backward compatibility
- `RunController.__init__` reads a manifest **exactly as written**; a v6 manifest is not
  auto-migrated and its on-disk `schema_version` is **not** bumped in place.
- v7 accessor methods default the additive fields when absent, so v7 code operates on a v6
  manifest in memory. If a v7 feature (e.g. `record_runtime_attempt`) is used on a v6 run and the
  run is saved, the additive field is persisted but `schema_version` stays 6 — the additive keys
  are ignored by v6-only code.

## Migration (copy only — never in place)
```python
from workflow.manifest_migration import migrate_run_manifest
migrate_run_manifest("runs/old_v6", "runs/old_v6_v7")   # source untouched; copy becomes v7
```
- The source directory (including a frozen baseline) is never modified.
- On any error (existing destination, missing/invalid/newer manifest), the partial copy is
  removed and the original is preserved.
- A fresh run created by `RunController.initialize` is v7 directly.

## Rollback
- Because migration is copy-only, rollback = **discard the migrated copy and keep using the v6
  original**. Nothing to undo on the source.
- To force a v7 manifest back to v6: delete the `runtime_attempts` and `idempotency` keys (and any
  `runner` objects) and set `schema_version: 6`. This is only safe when no v7-only feature state
  is being relied upon; prefer keeping the original v6 copy.
