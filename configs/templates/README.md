# Generic run-config skeletons

These files describe interfaces, not a specific model or material. The Director
copies them to `configs/runs/<run>/`, fills only reviewed values, and selects
built-in or external adapter callables. Case-specific configs remain under
`examples/` or `configs/examples/`.

`null` values are intentional blockers. Preflight must reject unresolved
scientific settings before a pilot.
