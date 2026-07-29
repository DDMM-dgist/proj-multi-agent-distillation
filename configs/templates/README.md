# Generic run-config skeletons

These files describe interfaces, not a specific model or material. The Director
copies them to `configs/runs/<run>/`, fills only reviewed values, and selects
built-in or external adapter callables. Case-specific configs remain under
`examples/` or `configs/examples/`.

`null` values are intentional blockers. Preflight must reject unresolved
scientific settings before a pilot.

`workflow.yaml` includes the full generic route through uncertainty,
deployment MD, optional reference validation, physical validation, and final
analysis. Remove an optional stage only after the run scope records why it is
not applicable; otherwise replace its `null` command/contract with the selected
adapter and evidence contract.
