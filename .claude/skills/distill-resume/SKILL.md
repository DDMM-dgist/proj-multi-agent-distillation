---
name: distill-resume
description: Resume an initialized distillation run from its persistent manifest after reviewing its last state.
argument-hint: "[run name or runs/path]"
disable-model-invocation: false
---

Locate and read the run manifest, workflow snapshot, latest logs, registered
artifacts, and gate evidence. Verify that files referenced by the current stage
still exist. Explain where the run stopped and why.

- If waiting for a scientific or cost approval, ask for it and do not execute.
- If a stage failed, propose the smallest repair and wait when it changes
  scientific settings or cost.
- If an intentionally edited declared input is blocking execution, summarize
  its old and new hash and obtain researcher approval before using
  `workflow.controller rebind-inputs`. This invalidates all earlier stage
  results while preserving the prior audit trail.
- If a producer artifact is complete but unregistered, verify it before using
  `complete-stage`.
- If a gate is pending, convene the judge committee; do not silently mark PASS.
- If the next stage is inexpensive and already approved, continue it through
  the controller.

Resume an intact run when its bound inputs and code are unchanged. If the code
revision changed, explain why the controller blocks continuation and propose a
new, explicitly linked run; do not silently treat it as the old run.
