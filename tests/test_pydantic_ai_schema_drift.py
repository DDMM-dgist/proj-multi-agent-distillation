"""Drift tests: the Pydantic typed-input mirror must stay in lockstep with the canonical
JSON Schema, and a successful mirror parse must NOT bypass the authoritative validator.

Network-free. Skips cleanly when the optional ``pydantic`` extra is absent.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "orchestration" / "schema"

try:  # optional dependency, mirrors tests/test_pydantic_ai_runtime.py
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class AgentTaskModelDriftTests(unittest.TestCase):
    def setUp(self):
        from runtimes.pydantic_ai.models import AgentTaskModel

        self.model = AgentTaskModel
        self.schema = _load_schema("agent_task.schema.json")
        self.fields = self.model.model_fields
        self.valid = {
            "schema_version": 1,
            "task_id": "t-1",
            "agent": "judge",
            "created_at": "2026-08-06T00:00:00Z",
            "instruction": "review the evidence",
            "inputs": [{"role": "data-curator", "path": "runs/x/coverage.json"}],
            "criteria": ["c1"],
            "constraints": ["k1"],
            "context": {"review_lens": "evidence_provenance", "review_focus": "provenance"},
        }

    def test_required_fields_match_canonical(self):
        canonical_required = set(self.schema["required"])
        model_required = {n for n, f in self.fields.items() if f.is_required()}
        self.assertEqual(
            model_required, canonical_required,
            f"required drift: model={sorted(model_required)} schema={sorted(canonical_required)}")

    def test_property_set_matches_canonical(self):
        canonical_props = set(self.schema["properties"])
        model_props = set(self.fields)
        self.assertEqual(model_props, canonical_props,
                         f"property drift: model={sorted(model_props)} schema={sorted(canonical_props)}")

    def test_run_id_is_the_only_optional(self):
        optional = {n for n, f in self.fields.items() if not f.is_required()}
        self.assertEqual(optional, {"run_id"})

    def test_extra_fields_rejected(self):
        import pydantic
        with self.assertRaises(pydantic.ValidationError):
            self.model(**{**self.valid, "surprise": True})

    def test_schema_version_is_const_1(self):
        import pydantic
        # canonical uses {"const": 1}
        self.assertEqual(self.schema["properties"]["schema_version"], {"const": 1})
        with self.assertRaises(pydantic.ValidationError):
            self.model(**{**self.valid, "schema_version": 2})

    def test_empty_strings_rejected_for_non_empty_fields(self):
        import pydantic
        for field in ("task_id", "agent", "created_at", "instruction"):
            with self.assertRaises(pydantic.ValidationError):
                self.model(**{**self.valid, field: ""})
        with self.assertRaises(pydantic.ValidationError):
            self.model(**{**self.valid, "criteria": [""]})

    def test_valid_task_parses(self):
        parsed = self.model(**self.valid)
        self.assertEqual(parsed.task_id, "t-1")
        self.assertEqual(parsed.inputs[0].role, "data-curator")

    def test_missing_required_field_rejected(self):
        import pydantic
        for field in self.schema["required"]:
            payload = {k: v for k, v in self.valid.items() if k != field}
            with self.assertRaises(pydantic.ValidationError):
                self.model(**payload)

    def test_parse_success_does_not_bypass_validate_task(self):
        """A Judge task that parses fine as AgentTaskModel but lacks the Judge-required
        context (review_lens/review_focus) must still be rejected by validate_task."""
        from orchestration.exchange import validate_task
        from orchestration.specs import load_agent_specs

        specs = load_agent_specs(str(ROOT / "agent_specs"))
        judge_spec = specs["judge"]
        bad = {**self.valid, "context": {}}  # parses as model, but Judge needs review_lens/focus
        # It parses as the typed mirror:
        self.model(**bad)
        # ...yet the authoritative validator rejects it:
        with self.assertRaises(ValueError):
            validate_task(bad, judge_spec)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
