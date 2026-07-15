---
name: distill-status
description: Show the current stage, gates, artifacts, failures, and next decision for a distillation run.
argument-hint: "[run name or runs/path]"
disable-model-invocation: false
---

Locate the requested run, or the only active run if unambiguous. Read its
`manifest.json`, workflow snapshot, latest stage logs, and gate evidence. Run
`python -m workflow.controller status <run_dir>`. Report in plain language:

- completed and currently blocked stages;
- latest PASS/REVISE/FAIL decisions;
- important artifacts already available;
- any failed command and its concise cause;
- the single next action or researcher decision needed.

Do not start or rerun work unless the researcher asks.
