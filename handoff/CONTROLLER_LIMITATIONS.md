# Known controller limitation (handoff)

## CONTROLLER_POSTHOC_ARTIFACT_ADOPTION = UNSUPPORTED
When a stage's gate is recorded REVISE/FAIL (even a false-negative), the native recovery flow
(`propose-recovery` → `approve-recovery` → `start-iteration`) **invalidates from the return stage and
forces re-execution**. There is no native command to *supersede* a false-negative gate by ADOPTING the
existing, already-validated artifact without re-running the stage. `complete-external-stage` can adopt
externally-produced artifacts, but only when `pending_recovery` is cleared and the previous gate is
PASS — so it cannot bridge an upstream false-negative REVISE.

**Consequence for handoff:** a gate helper false-negative on an expensive stage cannot be reconciled
natively without recomputation. In this campaign it was reconciled via an **audited operator state
closure** (original REVISE event preserved; superseding recovery event `FALSE_NEGATIVE_SUPERSEDED`;
existing artifacts deterministically re-validated; `manifest.pre-closure.json` backup) with **no compute
rerun**.

**Recommended future enhancement (not made here — frozen core):** a `supersede-gate`/`adopt-artifacts`
controller command that, given an unchanged/validated artifact set and a corrected deterministic check,
records a SUPERSEDING_PASS and registers downstream artifacts without re-execution — preserving the
original failure event.
