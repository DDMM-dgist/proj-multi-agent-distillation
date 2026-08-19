"""Regression: LLM-facing bounded-evidence must compactly represent large directory artifacts.

Reproduces the R31 training-gate defect where the declared ``artifacts/committee/`` output (four
seeds, ~1,500 files) expanded through ``artifact_digest``'s complete per-file listing and pushed
``build_bounded_evidence`` past ``MAX_EVIDENCE_BYTES`` (256 KiB). The fix compacts ONLY the
evidence representation (``summarize_artifact``); canonical ``artifact_digest`` integrity hashing
is unchanged.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflow.integrity import artifact_digest
from runtimes.pydantic_ai.bounded_evidence import (
    EVIDENCE_DIRECTORY_FILE_CAP,
    MAX_EVIDENCE_BYTES,
    build_bounded_evidence,
    summarize_artifact,
)


def _make_big_dir(root: Path, n_files: int) -> Path:
    d = root / "committee"
    (d / "data").mkdir(parents=True)
    # Deterministic, zero-padded names so sorted-path order is well-defined.
    for i in range(n_files):
        (d / "data" / f"data{i:05d}.pt").write_bytes(f"payload-{i}".encode())
    return d


class DirectorySummaryBoundTests(unittest.TestCase):
    def test_large_directory_summarizes_below_limit(self):
        # A: a directory with ~1,500 files summarizes well below MAX_EVIDENCE_BYTES.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = _make_big_dir(root, 1500)
            summary = summarize_artifact(big)
            nbytes = len(json.dumps(summary, indent=2, sort_keys=True).encode())
            self.assertLess(nbytes, MAX_EVIDENCE_BYTES)

            integ = summary["integrity"]
            # B: required aggregate fields are preserved.
            self.assertEqual(integ["kind"], "directory")
            self.assertIn("sha256", integ)
            self.assertIn("size", integ)
            self.assertEqual(integ["n_files"], 1500)
            self.assertEqual(len(integ["files_shown"]), EVIDENCE_DIRECTORY_FILE_CAP)
            self.assertEqual(
                integ["n_files_omitted"], 1500 - EVIDENCE_DIRECTORY_FILE_CAP)
            # The compact evidence must NOT inline the full listing.
            self.assertNotIn("files", integ)

    def test_full_bundle_with_manifest_below_limit(self):
        # A/end-to-end: manifest + big dir through build_bounded_evidence no longer raises.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = _make_big_dir(root, 1500)
            manifest = root / "committee.manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "models": [1, 2, 3, 4]}))
            out = root / "evidence.json"
            payload = build_bounded_evidence([manifest, big], out)
            self.assertLess(len(out.read_text().encode()), MAX_EVIDENCE_BYTES)
            self.assertEqual(len(payload["artifacts"]), 2)

    def test_ordering_is_deterministic(self):
        # C: the shown subset is stable across repeated summarization and matches sorted
        # relative-path order.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = _make_big_dir(root, 200)
            first = summarize_artifact(big)["integrity"]["files_shown"]
            second = summarize_artifact(big)["integrity"]["files_shown"]
            self.assertEqual(first, second)
            paths = [f["path"] for f in first]
            self.assertEqual(paths, sorted(paths))

    def test_canonical_artifact_digest_unchanged(self):
        # D: canonical artifact_digest still emits the COMPLETE per-file listing (integrity
        # semantics untouched by the evidence-only compaction).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = _make_big_dir(root, 1500)
            digest = artifact_digest(big)
            self.assertEqual(digest["kind"], "directory")
            self.assertEqual(len(digest["files"]), 1500)
            self.assertNotIn("files_shown", digest)
            self.assertNotIn("n_files_omitted", digest)

    def test_small_directory_keeps_full_listing(self):
        # E (dir variant): a directory at/below the cap keeps every file, nothing omitted.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = _make_big_dir(root, 5)
            integ = summarize_artifact(small)["integrity"]
            self.assertEqual(integ["n_files"], 5)
            self.assertEqual(len(integ["files_shown"]), 5)
            self.assertEqual(integ["n_files_omitted"], 0)

    def test_small_file_artifact_summarizes_normally(self):
        # E: ordinary small file artifacts are unaffected (file-kind integrity, no dir fields).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "x.json"
            f.write_text(json.dumps({"schema_version": 1, "status": "PASS"}))
            summary = summarize_artifact(f)
            self.assertEqual(summary["summary_kind"], "json_manifest")
            self.assertEqual(summary["integrity"]["kind"], "file")
            self.assertNotIn("files_shown", summary["integrity"])


if __name__ == "__main__":
    unittest.main()
