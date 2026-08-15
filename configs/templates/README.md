# Generic run-config skeletons

These files describe interfaces, not a specific model or material. The Orchestrator
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

`data_coverage`, `uncertainty`, `physical_validation`, and `analysis` already
carry their canonical `pydantic_ai` role/action pair and output contract
(registered in `runtimes/pydantic_ai/actions.py` / `executors.py::BINDINGS`) --
do not rename or invent a different action for these. Only their
`parameters: {}` are run-specific and must be filled in before initialization.
