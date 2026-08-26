"""FE-048 -- canonical protected-reference resolver for protection-consuming actions.

Demonstrated blocker (ffv4t-eng4, Stage 5): the ``teacher_labeling`` stage declares
``teacher_labeling_protection_audit.json`` as a gated output, but the runtime ``reference_yaml``
injection in ``_proposal_from_stage`` resolved ONLY a bound ``protected-existing-dft`` reference.
This campaign's canonical protection reference is the identity-only kind
``protected-structure-identity`` (FE-023), so ``reference_yaml`` was never injected, the executor
skipped the audit, and the stage failed "missing declared outputs" (exit 2) -- AFTER the real
Teacher labeling had already succeeded.

The fix unifies protection-consuming-action reference resolution through one canonical resolver
(``_canonical_protected_reference_yaml``) that resolves the bound protection reference generically
(``protected-existing-dft`` precedence, else ``protected-structure-identity``), applied consistently
to ``acquire_structures`` / ``label_with_teacher`` / ``train_committee``, and fails closed when a
stage that declares a protection audit output can resolve no canonical protected reference.

These tests prove, deterministically and without GPU/Teacher/network:

  1. a ``protected-structure-identity`` reference resolves for ``label_with_teacher``;
  2. a ``protected-existing-dft`` reference remains compatible (and takes precedence);
  3. no canonical protection reference + a declared protection audit output -> fail closed;
  4. the teacher_labeling protection audit is emitted from the resolved reference;
  5. a protected-population overlap in the labeled output fails closed (no audit written);
  6. ``acquisition`` and ``teacher_labeling`` resolve the IDENTICAL canonical protected population;
  7. the exact eng4 Stage-5 binding shape (identity-only protection + recovered-holdout bound,
     teacher_labeling declaring the audit with no reference_yaml param) receives a non-null
     reference_yaml.
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path


def _skip_without_optional_deps(test):
    try:
        import ase  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
        from runtimes.pydantic_ai import cli  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - core-only install
        test.skipTest(f"optional dep not installed: {exc}")


REQUIRED_PROHIBITIONS = [
    "student_training",
    "student_validation_tuning",
    "acquisition_seed",
    "augmentation_parent",
    "recovery_training",
]
PROTECTED_INDICES = [760, 761]


def _identity_frame(x):
    from ase import Atoms
    return Atoms("Si", positions=[[x, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)


def _labeled_frame(x, structure_id, parent, *, source_local_index, energy=1.0):
    """A labeled Student-side frame: geometry + workflow lineage + Teacher labels."""
    import numpy as np
    from ase import Atoms
    atoms = Atoms("SiO2", positions=[[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]],
                  cell=[8, 8, 8], pbc=True)
    atoms.info["structure_id"] = structure_id
    atoms.info["source_category"] = "bulk_amo"
    atoms.info["source_local_index"] = source_local_index
    atoms.info["parent_structure_id"] = parent
    atoms.info["teacher_energy"] = float(energy)
    atoms.info["label_source"] = "teacher"
    atoms.arrays["teacher_forces"] = np.zeros((len(atoms), 3))
    return atoms


def _build_structure_identity_reference(root):
    """A valid geometry-only ``protected-structure-identity`` reference.yaml (FE-023 kind)."""
    import yaml
    from ase.io import write
    from workflow.integrity import sha256_file
    from validation.protected_reference import PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS

    root = Path(root)
    indices_path = root / "protected_source_indices.txt"
    indices_path.write_text("\n".join(str(i) for i in PROTECTED_INDICES) + "\n")
    # Protected geometry sits far (x=50) from the labeled population (disjoint by construction).
    structures_path = root / "protected_geometry.extxyz"
    write(str(structures_path), [_identity_frame(50.0), _identity_frame(51.0)])
    reference = root / "protection_identity_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "protected-structure-identity",
        "reference_id": "fe048-protection-only",
        "reference_class": PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS,
        "status": "IDENTITY_AVAILABLE_AND_PROTECTED",
        "protected_source_rows": len(PROTECTED_INDICES),
        "protected_source_indices_file": str(indices_path),
        "protected_source_indices_sha256": sha256_file(indices_path),
        "structures": {
            "path": str(structures_path),
            "logical_frames": 2,
            "sha256": sha256_file(structures_path),
        },
        "prohibited_uses": list(REQUIRED_PROHIBITIONS),
    }))
    return reference


def _build_existing_dft_reference(root):
    """A valid DFT-labeled ``protected-existing-dft`` EVALUATION reference.yaml."""
    from ase.io import write
    from workflow.integrity import sha256_file

    root = Path(root)
    protected = root / "protected_existing_dft.extxyz"
    write(str(protected), [_identity_frame(70.0)])
    rows = root / "protected_rows.txt"
    rows.write_text("760\n761\n", encoding="utf-8")
    rows_csv = root / "protected_source_rows.csv"
    rows_csv.write_text(
        "global_index,category,source_file_relative,source_local_index,natoms,config_type\n"
        "760,bulk_amo,bulk_amo/bulk_amo.xyz,660,1,bulk_amo\n"
        "761,bulk_amo,bulk_amo/bulk_amo.xyz,661,1,bulk_amo\n",
        encoding="utf-8",
    )
    manifest = root / "protected_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1, "matched_logical_frames": 1,
        "unmatched_logical_frames": 0, "protected_source_rows": 2,
        "conflicting_label_duplicates": 0,
    }}), encoding="utf-8")
    ref = root / "existing_dft_reference.yaml"
    ref.write_text("\n".join([
        "kind: protected-existing-dft",
        "reference_id: fe048-existing-dft",
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
        "prohibited_uses: [student_training, student_validation_tuning, acquisition_seed, "
        "augmentation_parent, recovery_training]",
        "structures:",
        f"  path: {protected}",
        "  logical_frames: 1",
        f"  sha256: {sha256_file(protected)}",
        "",
    ]), encoding="utf-8")
    return ref


def _teacher_labeling_stage(root, *, with_reference_param=False):
    """The teacher_labeling stage config -- declaring the protection audit output. Mirrors the
    real workflow.yaml: NO reference_yaml param (that is what the runtime must inject)."""
    params = {
        "teacher_config": str(root / "teacher.yaml"),
        "structures_path": str(root / "labeled.extxyz"),
        "out_path": str(root / "artifacts" / "teacher_labeled.extxyz"),
        "manifest_path": str(root / "artifacts" / "teacher_labels.manifest.json"),
        "include_stress": False,
        "selected_source_indices": [],
    }
    if with_reference_param:
        params["reference_yaml"] = str(root / "protection_identity_reference.yaml")
    return {
        "name": "teacher_labeling",
        "outputs": ["artifacts/teacher_labeled.extxyz",
                    "artifacts/teacher_labels.manifest.json",
                    "artifacts/teacher_labeling_protection_audit.json"],
        "pydantic_ai": {"role": "data-curator", "action": "label_with_teacher",
                        "parameters": params},
    }


def _acquisition_stage(root, reference_yaml):
    return {
        "name": "acquisition",
        "outputs": ["artifacts/acquisition_candidates.extxyz",
                    "artifacts/acquisition.manifest.json",
                    "artifacts/acquisition_protection_audit.json"],
        "pydantic_ai": {"role": "data-curator", "action": "acquire_structures",
                        "parameters": {
                            "acquisition_config": str(root / "acq.yaml"),
                            "teacher_config": str(root / "teacher.yaml"),
                            "out_path": str(root / "artifacts" / "acquisition_candidates.extxyz"),
                            "manifest_path": str(root / "artifacts" / "acquisition.manifest.json"),
                            "selected_source_indices": [0, 1],
                            "reference_yaml": str(Path(reference_yaml).resolve()),
                        }},
    }


def _controller(root, *, input_refs, stages):
    import yaml
    root = Path(root)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    wf_path = root / "workflow.yaml"
    wf_path.write_text(yaml.safe_dump({"stages": stages}), encoding="utf-8")
    state = {
        "inputs": [{"source": str(Path(r).resolve())} for r in input_refs],
        "workflow_config": str(wf_path),
        "project_dir": str(root),
        "run_id": "fe048-test",
        "iterations": [{"id": 1}],
    }
    by_name = {s["name"]: s for s in stages}
    return types.SimpleNamespace(run_dir=root, state=state, stage=lambda n: by_name[n])


def _reference_yaml_of(proposal_tuple):
    return (proposal_tuple[0].get("parameters") or {}).get("reference_yaml")


class FE048CanonicalProtectedReferenceResolver(unittest.TestCase):
    def setUp(self):
        _skip_without_optional_deps(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # ---- 1: protected-structure-identity resolves for label_with_teacher --------------------
    def test_structure_identity_resolves_for_label_with_teacher(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        ref = _build_structure_identity_reference(self.root)
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[ref], stages=[stage])
        prop = _proposal_from_stage(c, "teacher_labeling", stage)
        self.assertEqual(_reference_yaml_of(prop), str(ref.resolve()))

    # ---- 2: protected-existing-dft remains compatible (and takes precedence) ----------------
    def test_existing_dft_takes_precedence_and_remains_compatible(self):
        from runtimes.pydantic_ai.cli import (
            _proposal_from_stage, _canonical_protected_reference_yaml)
        identity = _build_structure_identity_reference(self.root)
        existing = _build_existing_dft_reference(self.root)
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[identity, existing], stages=[stage])
        # existing-dft has precedence over identity-only when both are bound.
        self.assertEqual(_canonical_protected_reference_yaml(c), str(existing.resolve()))
        prop = _proposal_from_stage(c, "teacher_labeling", stage)
        self.assertEqual(_reference_yaml_of(prop), str(existing.resolve()))

    # ---- 3: no canonical protection reference + declared audit -> fail closed ----------------
    def test_no_protection_reference_fails_closed(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[], stages=[stage])
        with self.assertRaises(ValueError) as ctx:
            _proposal_from_stage(c, "teacher_labeling", stage)
        self.assertIn("protection audit", str(ctx.exception))

    # ---- 4: teacher_labeling audit emitted from the resolved reference ----------------------
    def test_teacher_labeling_audit_emitted_via_resolved_reference(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        from runtimes.pydantic_ai import executors
        from validation.protected_reference import validate_protection_audit_report
        from ase.io import write
        import adapters
        import adapters.acquisition as acq
        from workflow.integrity import sha256_file

        ref = _build_structure_identity_reference(self.root)
        # Disjoint labeled population: distinct geometry, non-protected rows + lineage.
        labeled = [_labeled_frame(0.0, "cand:0", "seed-pool:1", source_local_index=1),
                   _labeled_frame(4.0, "cand:1", "seed-pool:2", source_local_index=2)]
        labeled_path = self.root / "labeled.extxyz"
        write(str(labeled_path), labeled)
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[ref], stages=[stage])
        prop_params = _proposal_from_stage(c, "teacher_labeling", stage)[0]["parameters"]
        # structures_path is the labeled population under audit.
        prop_params["structures_path"] = str(labeled_path)
        out_path = Path(prop_params["out_path"])
        manifest_path = Path(prop_params["manifest_path"])

        def _fake_label(_cfg, structures_path, out, manifest_p, _stress):
            write(str(out), labeled)
            obj = {"schema_version": 1, "output": str(Path(out).resolve()),
                   "sha256": sha256_file(Path(out)), "n_frames": len(labeled)}
            Path(manifest_p).write_text(json.dumps(obj), encoding="utf-8")
            return obj

        orig_label, orig_cfg = acq.label_with_teacher, adapters.load_config
        acq.label_with_teacher = _fake_label
        adapters.load_config = lambda _p: {"stub": True}
        self.addCleanup(lambda: setattr(acq, "label_with_teacher", orig_label))
        self.addCleanup(lambda: setattr(adapters, "load_config", orig_cfg))

        result = executors._exec_label_with_teacher({"parameters": prop_params})
        audit_path = out_path.with_name("teacher_labeling_protection_audit.json")
        self.assertTrue(audit_path.exists(), "protection audit must be emitted")
        report = json.loads(audit_path.read_text())
        self.assertEqual(report["stage"], "teacher_labeling")
        validate_protection_audit_report(
            audit_path, reference_yaml=str(ref),
            submitted_artifacts=[audit_path.resolve(), out_path.resolve()])
        self.assertEqual(result["protection_audit_path"], str(audit_path.resolve()))
        self.assertIn("protection_audit_integrity", result)
        self.assertTrue(manifest_path.exists())

    # ---- 5: protected-population overlap in the labeled output -> fail closed ----------------
    def test_protected_overlap_fails_closed(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        from runtimes.pydantic_ai import executors
        from ase.io import write
        import adapters
        import adapters.acquisition as acq
        from workflow.integrity import sha256_file

        import numpy as np
        ref = _build_structure_identity_reference(self.root)
        # A labeled frame whose GEOMETRY is byte-identical to a protected reference frame
        # (single Si at x=50, same cell/pbc) -- labeling only attaches Teacher fields, so the
        # geometry fingerprint still collides and protection must fail closed.
        leaking_atoms = _identity_frame(50.0)
        leaking_atoms.info["structure_id"] = "leak:0"
        leaking_atoms.info["source_category"] = "bulk_amo"
        leaking_atoms.info["source_local_index"] = 9
        leaking_atoms.info["parent_structure_id"] = "seed-pool:9"
        leaking_atoms.info["teacher_energy"] = 1.0
        leaking_atoms.info["label_source"] = "teacher"
        leaking_atoms.arrays["teacher_forces"] = np.zeros((len(leaking_atoms), 3))
        leaking = [leaking_atoms]
        labeled_path = self.root / "labeled.extxyz"
        write(str(labeled_path), leaking)
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[ref], stages=[stage])
        prop_params = _proposal_from_stage(c, "teacher_labeling", stage)[0]["parameters"]
        prop_params["structures_path"] = str(labeled_path)

        def _fake_label(_cfg, structures_path, out, manifest_p, _stress):
            write(str(out), leaking)
            obj = {"schema_version": 1, "output": str(Path(out).resolve()),
                   "sha256": sha256_file(Path(out)), "n_frames": len(leaking)}
            Path(manifest_p).write_text(json.dumps(obj), encoding="utf-8")
            return obj

        orig_label, orig_cfg = acq.label_with_teacher, adapters.load_config
        acq.label_with_teacher = _fake_label
        adapters.load_config = lambda _p: {"stub": True}
        self.addCleanup(lambda: setattr(acq, "label_with_teacher", orig_label))
        self.addCleanup(lambda: setattr(adapters, "load_config", orig_cfg))

        with self.assertRaises(Exception):
            executors._exec_label_with_teacher({"parameters": prop_params})
        audit_path = (self.root / "artifacts" / "teacher_labeling_protection_audit.json")
        self.assertFalse(audit_path.exists(),
                         "no audit may be written for a leaking population")

    # ---- 6: acquisition and teacher_labeling resolve the SAME canonical population -----------
    def test_acquisition_and_teacher_labeling_resolve_same_population(self):
        from runtimes.pydantic_ai.cli import (
            _proposal_from_stage, _acquisition_protection_reference_yaml,
            _canonical_protected_reference_yaml)
        ref = _build_structure_identity_reference(self.root)
        acq_stage = _acquisition_stage(self.root, ref)
        tl_stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[ref], stages=[acq_stage, tl_stage])
        self.assertEqual(
            _acquisition_protection_reference_yaml(c),
            _canonical_protected_reference_yaml(c))
        acq_ref = _reference_yaml_of(_proposal_from_stage(c, "acquisition", acq_stage))
        tl_ref = _reference_yaml_of(_proposal_from_stage(c, "teacher_labeling", tl_stage))
        self.assertEqual(acq_ref, tl_ref)
        self.assertEqual(tl_ref, str(ref.resolve()))

    # ---- 7: exact eng4 Stage-5 binding shape -> non-null reference_yaml ----------------------
    def test_eng4_stage5_fixture_receives_non_null_reference_yaml(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage
        # eng4 bound BOTH an identity-only protection reference AND a recovered-original-holdout
        # (evaluation) reference; teacher_labeling declared the audit with no reference_yaml param.
        identity = _build_structure_identity_reference(self.root)
        holdout = _build_recovered_holdout_reference(self.root)
        stage = _teacher_labeling_stage(self.root)
        c = _controller(self.root, input_refs=[identity, holdout], stages=[stage])
        ref = _reference_yaml_of(_proposal_from_stage(c, "teacher_labeling", stage))
        self.assertIsNotNone(ref)
        # The recovered-holdout (evaluation-only) is never substituted for the protected population.
        self.assertEqual(ref, str(identity.resolve()))


def _build_recovered_holdout_reference(root):
    import numpy as np
    import yaml
    from ase import Atoms
    from ase.io import write
    from workflow.integrity import sha256_file
    from validation.protected_reference import (
        RECOVERED_HOLDOUT_REFERENCE_CLASS, RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS)

    root = Path(root)
    records = [
        {"source_category": "bulk", "source_local_index": 0, "split": "train"},
        {"source_category": "bulk", "source_local_index": 1, "split": "test"},
        {"source_category": "bulk", "source_local_index": 2, "split": "test"},
    ]
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(json.dumps({"records": records}))

    def frame(local_index, x):
        a = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["source_category"] = "bulk"
        a.info["source_local_index"] = local_index
        a.info["dft_energy"] = -1.0
        a.arrays["dft_forces"] = np.array([[0.0, 0.0, 0.0]])
        return a

    frames = [frame(1, 1.0), frame(2, 2.0)]
    structures_path = root / "holdout.extxyz"
    write(str(structures_path), frames)
    reference = root / "recovered_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "recovered-original-holdout",
        "reference_id": "fe048-recovered-holdout",
        "reference_class": RECOVERED_HOLDOUT_REFERENCE_CLASS,
        "status": "AVAILABLE_AND_VERIFIED",
        "target_split": "test",
        "split_source_manifest": str(manifest_path),
        "split_source_manifest_sha256": sha256_file(manifest_path),
        "frame_count": 2,
        "structures": {
            "path": str(structures_path),
            "logical_frames": 2,
            "sha256": sha256_file(structures_path),
        },
        "prohibited_uses": list(RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS),
    }))
    return reference


if __name__ == "__main__":
    unittest.main()
