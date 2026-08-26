"""FE-058: the Student-distillation merge's source global-index offset derivation must
tolerate a PROTECTION-ONLY reference kind (e.g. ``protected-structure-identity``) that
declares no per-source-row CSV. Such a reference identifies protection by structure
fingerprints + explicit protected source indices, so there are simply no category
global-index offsets to derive -- the function must return an empty map instead of
crashing on the missing ``protected_source_rows_file`` key (the eng6 augmentation-merge
blocker). References that DO carry the rows CSV/file must keep deriving offsets exactly.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflow.steps import _source_global_offsets_from_reference


class MergeOffsetProtectionOnlyReferenceTests(unittest.TestCase):
    def _write(self, d: Path, name: str, text: str) -> str:
        p = d / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_no_source_rows_declared_returns_empty_offsets(self):
        with TemporaryDirectory() as td:
            ref = self._write(Path(td), "reference.yaml",
                              "kind: protected-structure-identity\n"
                              "descendants_of_protected_structures: PROHIBITED\n")
            self.assertEqual(_source_global_offsets_from_reference(ref), {})

    def test_empty_reference_returns_empty_offsets(self):
        with TemporaryDirectory() as td:
            ref = self._write(Path(td), "reference.yaml", "")
            self.assertEqual(_source_global_offsets_from_reference(ref), {})

    def test_rows_csv_still_derives_offsets(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            csv = self._write(
                d, "protected_source_rows.csv",
                "category,global_index,source_local_index\n"
                "liquid,5569,0\n"
                "liquid,5570,1\n"
                "crystal,8000,0\n")
            ref = self._write(d, "reference.yaml",
                              f"kind: protected-existing-dft\n"
                              f"protected_source_rows_csv: {csv}\n")
            offsets = _source_global_offsets_from_reference(ref)
            self.assertEqual(offsets, {"liquid": 5569, "crystal": 8000})

    def test_rows_file_key_resolves_sibling_csv(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            self._write(
                d, "protected_source_rows.csv",
                "category,global_index,source_local_index\n"
                "liquid,5569,0\n")
            index_file = self._write(d, "protected_source_indices.txt", "5569\n")
            ref = self._write(d, "reference.yaml",
                              f"kind: protected-existing-dft\n"
                              f"protected_source_rows_file: {index_file}\n")
            self.assertEqual(
                _source_global_offsets_from_reference(ref), {"liquid": 5569})


if __name__ == "__main__":
    unittest.main()
