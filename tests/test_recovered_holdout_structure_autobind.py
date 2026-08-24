"""Regression: a fresh run that binds a `recovered-original-holdout` reference must also
canonically hash-bind the structure population that reference points at, so the campaign can
reach reference_validation without a human noticing and manually calling bind_new_input mid-run.

The generic fix lives in workflow.controller: referenced_evidence_structure() +
EVIDENCE_STRUCTURE_REFERENCE_KINDS, wired into RunController.initialize() and
RunController.bind_new_input(). These tests start from an EMPTY synthetic run (no SiO2-specific
fixtures) and prove the invariant and its fail-closed edges.
"""
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController, referenced_evidence_structure
from workflow.integrity import artifact_digest
from validation.report import evidence_record, validate_evidence


def _write_structures(path, body="stub-structure-population\n"):
    path.write_text(body)
    return path


def _reference_yaml(structures_path, *, kind="recovered-original-holdout",
                    declared_sha=None, extra=None):
    doc = {
        "kind": kind,
        "reference_id": "recovered-original-heldout-test",
        "target_split": "test",
        "frame_count": 1142,
        "structures": {"path": str(structures_path)},
    }
    if declared_sha is not None:
        doc["structures"]["sha256"] = declared_sha
    if extra:
        doc.update(extra)
    return doc


def _minimal_cfg(run_id, inputs):
    return {"run_id": run_id, "inputs": inputs,
            "stages": [{"name": "reference_validation", "command": None}]}


class RecoveredHoldoutStructureAutobindTests(unittest.TestCase):
    def test_fresh_init_autobinds_reference_and_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = _write_structures(root / "recovered_original_holdout_test.xyz")
            structures_digest = artifact_digest(structures)
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump(
                _reference_yaml(structures, declared_sha=structures_digest["sha256"])))
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump(_minimal_cfg("holdout-fresh", [str(reference)])))

            controller = RunController.initialize(cfg, root / "run")

            # Reference YAML AND the structure population are both registered inputs.
            self.assertEqual(len(controller.state["inputs"]), 2)
            sources = {Path(r["source"]).resolve() for r in controller.state["inputs"]}
            self.assertIn(reference.resolve(), sources)
            self.assertIn(structures.resolve(), sources)

            structure_record = next(r for r in controller.state["inputs"]
                                    if Path(r["source"]).resolve() == structures.resolve())
            # Hash-bound in place (copy=False, no snapshot), sha256 recorded from the real file.
            self.assertFalse(structure_record["copy"])
            self.assertIsNone(structure_record["snapshot"])
            self.assertEqual(structure_record["sha256"], structures_digest["sha256"])

            # The structure file is in the validation-evidence allowlist.
            allowlist = controller._validation_evidence_allowlist([], "reference_validation")
            self.assertIn(structures.resolve(), set(allowlist))

            # Input verification passes (no code/input drift).
            controller.verify_inputs()

            # And the canonical evidence gate does NOT reject the structure as unbound.
            manifest = root / "report.json"
            manifest.write_text("{}")
            roles = validate_evidence(
                manifest, [evidence_record("protected_reference_structures", structures)],
                allowed_evidence=allowlist, label="reference validation")
            self.assertEqual(roles, {"protected_reference_structures"})

    def test_already_declared_structure_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = _write_structures(root / "holdout.xyz")
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump(_reference_yaml(structures)))
            cfg = root / "workflow.yaml"
            # Declare BOTH the reference and the structure (copy=False) explicitly.
            cfg.write_text(yaml.safe_dump(_minimal_cfg(
                "holdout-predeclared",
                [str(reference), {"path": str(structures), "copy": False}])))

            controller = RunController.initialize(cfg, root / "run")

            self.assertEqual(len(controller.state["inputs"]), 2)
            structure_records = [r for r in controller.state["inputs"]
                                 if Path(r["source"]).resolve() == structures.resolve()]
            self.assertEqual(len(structure_records), 1)

    def test_missing_structure_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "does_not_exist.xyz"
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump(_reference_yaml(missing)))
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump(_minimal_cfg("holdout-missing", [str(reference)])))

            with self.assertRaises(FileNotFoundError):
                RunController.initialize(cfg, root / "run")
            self.assertFalse((root / "run").exists())

    def test_structure_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = _write_structures(root / "holdout.xyz")
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump(
                _reference_yaml(structures, declared_sha="0" * 64)))
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump(_minimal_cfg("holdout-mismatch", [str(reference)])))

            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                RunController.initialize(cfg, root / "run")
            self.assertFalse((root / "run").exists())

    def test_unrelated_reference_kind_is_not_autobound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = _write_structures(root / "identity.xyz")
            reference = root / "reference.yaml"
            # A reference of a DIFFERENT kind that also declares a structures.path must NOT
            # have its structures file auto-trusted/auto-bound.
            reference.write_text(yaml.safe_dump(
                _reference_yaml(structures, kind="protected-structure-identity")))
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump(_minimal_cfg("identity-only", [str(reference)])))

            controller = RunController.initialize(cfg, root / "run")

            self.assertEqual(len(controller.state["inputs"]), 1)
            sources = {Path(r["source"]).resolve() for r in controller.state["inputs"]}
            self.assertNotIn(structures.resolve(), sources)
            self.assertIsNone(
                referenced_evidence_structure(reference, project_dir=root))

            # The unbound structure is still rejected by the evidence gate (allowlist unchanged).
            allowlist = controller._validation_evidence_allowlist([], "reference_validation")
            manifest = root / "report.json"
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "not bound to this run"):
                validate_evidence(
                    manifest, [evidence_record("x", structures)],
                    allowed_evidence=allowlist, label="reference validation")

    def test_bind_new_input_autobinds_structure_post_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = _write_structures(root / "holdout.xyz")
            structures_digest = artifact_digest(structures)
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump(
                _reference_yaml(structures, declared_sha=structures_digest["sha256"])))
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump(_minimal_cfg("holdout-postinit", [])))

            controller = RunController.initialize(cfg, root / "run")
            self.assertEqual(len(controller.state["inputs"]), 0)

            controller.bind_new_input(reference)

            self.assertEqual(len(controller.state["inputs"]), 2)
            sources = {Path(r["source"]).resolve() for r in controller.state["inputs"]}
            self.assertIn(structures.resolve(), sources)
            auto_event = next(e for e in controller.state["events"]
                              if e.get("auto_bound_structures_for"))
            self.assertEqual(Path(auto_event["auto_bound_structures_for"]).resolve(),
                             reference.resolve())

            allowlist = controller._validation_evidence_allowlist([], "reference_validation")
            self.assertIn(structures.resolve(), set(allowlist))


if __name__ == "__main__":
    unittest.main()
