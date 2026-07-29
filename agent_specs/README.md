# Agent specifications

Each YAML file is the machine-readable registration record for one canonical
role. The corresponding natural-language instructions remain in `agents/*.md`.
Runtime-specific wrappers must reference these files; they must not duplicate or
silently alter the scientific role.

Validate the registry with:

```bash
python -m orchestration.cli validate-specs
```

The specification describes semantic capabilities, accepted/returned contract
types, approval boundaries, and delegation. It intentionally contains no model
name, provider API, or vendor-specific tool identifier.
