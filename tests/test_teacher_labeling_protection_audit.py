"""Regression test for the teacher_labeling protection-audit output.

Demonstrated blocker (C12): the ``teacher_labeling`` stage declares
``teacher_labeling_protection_audit.json`` as a gated output (a ``validation_manifest``
contract re-verifies it), but ``_exec_label_with_teacher`` previously ran only the
in-memory ``_protect_dataset`` check and never persisted the audit artifact, so the
campaign failed with "stage missing declared outputs".

The fix persists + validates the protected-reference exclusion audit for the labeled
output. These tests pin, without touching real Teacher inference (``label_with_teacher``
is monkeypatched):

  1. the audit artifact is written next to the manifest and validates for stage
     ``teacher_labeling``;
  2. the executor result surfaces ``protection_audit_path`` + integrity digest;
  3. a labeled dataset that collides with a protected reference fails closed before any
     audit is written.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from runtimes.pydantic_ai import executors
from validation.protected_reference import validate_protection_audit_report
from workflow.integrity import sha256_file


def _frame(x, structure_id, parent=None, energy=1.0, force_value=0.0):
    atoms = Atoms("SiO2", positions=[[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]],
                  cell=[8, 8, 8], pbc=True)
    atoms.info["structure_id"] = structure_id
    atoms.info["source_category"] = "bulk_amo"
    atoms.info["source_local_index"] = 0
    atoms.info["teacher_energy"] = float(energy)
    atoms.info["label_source"] = "teacher"
    if parent is not None:
        atoms.info["parent_structure_id"] = parent
    atoms.arrays["teacher_forces"] = np.full((len(atoms), 3), float(force_value))
    return atoms


class TeacherLabelingProtectionAuditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cfg = self.root / "cfg"
        self.cfg.mkdir()
        self.arts = self.root / "artifacts"
        self.arts.mkdir()
        # Protected reference geometry sits far from the labeled population (disjoint).
        self.protected_frame = _frame(50, "protected", energy=9.0)
        self.reference = self._reference()

    def _reference(self):
        protected = self.cfg / "protected.extxyz"
        write(protected, [self.protected_frame])
        rows = self.cfg / "protected_rows.txt"
        rows.write_text("760\n761\n", encoding="utf-8")
        rows_csv = self.cfg / "protected_source_rows.csv"
        rows_csv.write_text(
            "global_index,category,source_file_relative,source_local_index,natoms,config_type\n"
            "760,bulk_amo,bulk_amo/bulk_amo.xyz,660,3,bulk_amo\n"
            "761,bulk_amo,bulk_amo/bulk_amo.xyz,661,3,bulk_amo\n",
            encoding="utf-8",
        )
        manifest = self.cfg / "protected_manifest.json"
        manifest.write_text(json.dumps({"mapping": {
            "logical_test_frames": 1,
            "matched_logical_frames": 1,
            "unmatched_logical_frames": 0,
            "protected_source_rows": 2,
            "conflicting_label_duplicates": 0,
        }}), encoding="utf-8")
        ref = self.cfg / "reference.yaml"
        ref.write_text("\n".join([
            "kind: protected-existing-dft",
            "reference_id: fixture-reference",
            "reference_class: ORIGINAL_TEACHER_TEST",
            "status: AVAILABLE_AND_PROTECTED",
            "logical_test_frames: 1",
            "protected_source_rows: 2",
            f"protection_manifest: {manifest}",
            f"protected_source_rows_file: {rows}",
            f"protected_source_rows_csv: {rows_csv}",
            "duplicate_equivalent:",
            "  source_global_indices: [760, 761]",
            "  label_conflict: false",
            "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, augmentation_parent, recovery_training]",
            "structures:",
            f"  path: {protected}",
            "  logical_frames: 1",
            f"  sha256: {sha256_file(protected)}",
            "",
        ]), encoding="utf-8")
        return ref

    def _proposal(self, labeled_frames):
        """Build the labeled output on disk and a proposal that a monkeypatched
        label_with_teacher will "produce" (it just re-reports the pre-written file)."""
        out_path = self.arts / "teacher_labeled.extxyz"
        write(out_path, labeled_frames)
        manifest_path = self.arts / "teacher_labels.manifest.json"

        def _fake_label_with_teacher(_cfg, structures_path, out, manifest_p, _stress):
            # The real adapter reads structures_path, runs Teacher, and writes out+manifest.
            # Here the labeled frames are already on disk; just emit the manifest the executor
            # reports on. Geometry/labels are untouched, mirroring real labeling.
            written = Path(out)
            manifest_obj = {
                "schema_version": 1, "output": str(written.resolve()),
                "sha256": sha256_file(written), "n_frames": len(labeled_frames),
            }
            Path(manifest_p).write_text(json.dumps(manifest_obj), encoding="utf-8")
            return manifest_obj

        executors.__dict__  # noqa: B018 - ensure module imported
        self._orig = None
        import adapters.acquisition as acq
        self._orig = acq.label_with_teacher
        acq.label_with_teacher = _fake_label_with_teacher
        self.addCleanup(lambda: setattr(acq, "label_with_teacher", self._orig))

        import adapters
        self._orig_cfg = adapters.load_config
        adapters.load_config = lambda _p: {"stub": True}
        self.addCleanup(lambda: setattr(adapters, "load_config", self._orig_cfg))

        return {"parameters": {
            "structures_path": str(out_path),
            "out_path": str(out_path),
            "manifest_path": str(manifest_path),
            "teacher_config": str(self.cfg / "teacher.yaml"),
            "reference_yaml": str(self.reference),
            "selected_source_indices": [],
            "include_stress": False,
        }}

    def test_audit_written_and_validates(self):
        labeled = [_frame(0, "cand:0", parent="seed-pool:1"),
                   _frame(4, "cand:1", parent="seed-pool:2")]
        proposal = self._proposal(labeled)
        result = executors._exec_label_with_teacher(proposal)

        audit_path = self.arts / "teacher_labeling_protection_audit.json"
        self.assertTrue(audit_path.exists(), "protection audit artifact must be persisted")

        report = json.loads(audit_path.read_text())
        self.assertEqual(report["stage"], "teacher_labeling")

        # The gate re-runs exactly this validator; it must pass on the written audit.
        validate_protection_audit_report(
            audit_path, reference_yaml=str(self.reference),
            submitted_artifacts=[audit_path.resolve(),
                                 (self.arts / "teacher_labeled.extxyz").resolve()])

        self.assertEqual(result["protection_audit_path"], str(audit_path.resolve()))
        self.assertIn("protection_audit_integrity", result)

    def test_leaking_dataset_fails_closed_before_audit(self):
        # A labeled frame whose geometry equals the protected reference -> disjointness fails.
        leaking = [_frame(50, "leak:0", parent="seed-pool:1")]
        proposal = self._proposal(leaking)
        with self.assertRaises(Exception):
            executors._exec_label_with_teacher(proposal)
        # Fail-closed: no audit artifact is left behind claiming a clean population.
        self.assertFalse((self.arts / "teacher_labeling_protection_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
