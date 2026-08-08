"""Duplicate-read guard: a repeated identical (tool, resolved-path) that already succeeded in the
SAME agent run is refused fail-closed and recorded (provenance-visible), instead of re-serving
identical content. General liveness/safety guard motivated by Stage C attempt-2's Judge that
re-read the same evidence 6x. Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "pydantic not installed")
class DuplicateReadGuardTests(unittest.TestCase):
    def _toolset(self, d):
        from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset
        return ReadOnlyToolset([str(d)])

    def test_second_identical_read_refused_and_recorded(self):
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); (d / "e.json").write_text('{"structure_count": 12}')
            ts = self._toolset(d)
            v1 = ts.read_json(str(d / "e.json"))
            self.assertEqual(v1["structure_count"], 12)        # first read succeeds
            with self.assertRaises(ToolAccessError):
                ts.read_json(str(d / "e.json"))                # identical repeat -> refused
            inv = ts.invocations
            self.assertTrue(inv[-2].ok)                         # first recorded ok
            self.assertFalse(inv[-1].ok)                        # duplicate recorded as refusal
            self.assertIn("DUPLICATE_READ", inv[-1].detail)     # provenance-visible + nudges output

    def test_different_path_still_reads_and_guard_is_per_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.json").write_text('{"x": 1}'); (d / "b.json").write_text('{"y": 2}')
            ts = self._toolset(d)
            ts.read_json(str(d / "a.json"))
            self.assertEqual(ts.read_json(str(d / "b.json"))["y"], 2)   # different path is fine
            # a FRESH toolset (new agent run) may read the same path again
            self.assertEqual(self._toolset(d).read_json(str(d / "a.json"))["x"], 1)

    def test_failed_read_is_not_marked_as_duplicate(self):
        # a read that never succeeded (outside allow-list) must not poison the guard
        from runtimes.pydantic_ai.tool_registry import ToolAccessError
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); (d / "e.json").write_text('{"x": 1}')
            ts = self._toolset(d)
            with self.assertRaises(ToolAccessError):
                ts.read_json("/etc/shadow")                    # refused: outside allow-list
            with self.assertRaises(ToolAccessError):
                ts.read_json("/etc/shadow")                    # still an allow-list refusal...
            # ...NOT a DUPLICATE_READ (it never succeeded)
            self.assertNotIn("DUPLICATE_READ", ts.invocations[-1].detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
