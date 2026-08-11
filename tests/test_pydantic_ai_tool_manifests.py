"""Phase 3/E: bounded read-only summary tools + role tool manifests.

Network-free; skips when the optional ``pydantic`` extra is absent.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = str(ROOT / "agent_specs")

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class ReadSummaryToolTests(unittest.TestCase):
    def _toolset(self, root):
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        return ReadOnlyToolset([str(root)])

    def test_csv_summary_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = ["a,b,c"] + [f"{i},{i*2},{i*3}" for i in range(30)]
            (d / "data.csv").write_text("\n".join(rows) + "\n")
            ts = self._toolset(d)
            summary = ts.read_csv_summary(str(d / "data.csv"))
            self.assertEqual(summary["columns"], ["a", "b", "c"])
            self.assertEqual(summary["n_columns"], 3)
            self.assertEqual(summary["n_rows"], 30)
            self.assertEqual(len(summary["head"]), 20)     # bounded head
            self.assertTrue(summary["truncated"])
            self.assertEqual(ts.invocations[-1].tool, "read_csv_summary")
            self.assertTrue(ts.invocations[-1].ok)

    def test_csv_summary_outside_allowlist_refused(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            (Path(other) / "x.csv").write_text("a\n1\n")
            ts = self._toolset(tmp)
            with self.assertRaises(ToolAccessError):
                ts.read_csv_summary(str(Path(other) / "x.csv"))
            self.assertFalse(ts.invocations[-1].ok)

    def test_artifact_manifest_accepts_manifest_and_refuses_plain_json(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "manifest.json").write_text(json.dumps({"schema_version": 1, "sha256": "abc"}))
            (d / "plain.json").write_text(json.dumps({"hello": "world"}))
            ts = self._toolset(d)
            m = ts.read_artifact_manifest(str(d / "manifest.json"))
            self.assertEqual(m["schema_version"], 1)
            self.assertTrue(ts.invocations[-1].ok)
            with self.assertRaises(ToolAccessError):
                ts.read_artifact_manifest(str(d / "plain.json"))
            self.assertFalse(ts.invocations[-1].ok)

    def test_new_tools_are_in_the_exposed_surface(self):
        from runtimes.pydantic_ai.tool_registry import EXPOSED_READ_TOOLS
        self.assertIn("read_csv_summary", EXPOSED_READ_TOOLS)
        self.assertIn("read_artifact_manifest", EXPOSED_READ_TOOLS)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RoleToolManifestTests(unittest.TestCase):
    def setUp(self):
        from runtimes.pydantic_ai.tool_manifests import ROLE_TOOL_MANIFESTS
        from orchestration.specs import load_agent_specs
        self.manifests = ROLE_TOOL_MANIFESTS
        self.specs = load_agent_specs(SPECS)

    def test_every_role_has_a_manifest(self):
        self.assertEqual(set(self.manifests), set(self.specs))

    def test_read_tool_surface_is_uniform_and_within_exposed(self):
        from runtimes.pydantic_ai.tool_registry import EXPOSED_READ_TOOLS
        for role, m in self.manifests.items():
            self.assertEqual(tuple(m.allowed_read_tools), EXPOSED_READ_TOOLS, role)

    def test_dangerous_capabilities_denied_for_every_role(self):
        from runtimes.pydantic_ai.tool_manifests import UNIVERSALLY_DENIED
        for role, m in self.manifests.items():
            for cap in ("Bash", "Write", "Edit", "Glob", "shell"):
                self.assertIn(cap, m.denied_tools, f"{role} must deny {cap}")
            self.assertEqual(set(m.denied_tools) & set(m.allowed_read_tools), set(), role)
            self.assertTrue(set(UNIVERSALLY_DENIED) <= set(m.denied_tools), role)

    def test_judge_is_read_only(self):
        m = self.manifests["judge"]
        self.assertEqual(m.role_type, "reviewer")
        self.assertEqual(m.proposable_actions, [])
        self.assertEqual(m.write_roots, [])
        self.assertEqual(m.side_effect_class, "none")
        # a Judge must not be able to see peers' votes or the aggregate verdict
        self.assertIn("read_other_judge_votes", m.denied_tools)
        self.assertIn("read_aggregate_verdict", m.denied_tools)

    def test_producers_expose_exactly_their_allowed_actions(self):
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        for role, allowed in ROLE_ALLOWED_ACTIONS.items():
            self.assertEqual(set(self.manifests[role].proposable_actions), set(allowed), role)

    def test_no_manifest_proposes_an_unavailable_action(self):
        from runtimes.pydantic_ai.actions import CAPABILITY_REGISTRY
        unavailable = set(CAPABILITY_REGISTRY)
        for role, m in self.manifests.items():
            self.assertEqual(set(m.proposable_actions) & unavailable, set(), role)

    def test_approval_required_matches_agent_specs(self):
        for role, m in self.manifests.items():
            spec_boundaries = set(getattr(self.specs[role], "approval_boundaries", []) or [])
            self.assertEqual(set(m.approval_required), spec_boundaries,
                             f"{role}: manifest {sorted(m.approval_required)} != "
                             f"spec {sorted(spec_boundaries)}")

    def test_orchestrator_has_typed_bridge_not_shell(self):
        m = self.manifests["orchestrator"]
        self.assertIn("dispatch_agent_task", m.bridge_actions)
        self.assertIn("propose_gate_record", m.bridge_actions)
        self.assertIn("Bash", m.denied_tools)

    def test_manifests_serialize(self):
        from runtimes.pydantic_ai.tool_manifests import all_manifests_json
        blob = json.dumps(all_manifests_json())  # must be JSON-serializable
        self.assertIn("data-curator", blob)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
