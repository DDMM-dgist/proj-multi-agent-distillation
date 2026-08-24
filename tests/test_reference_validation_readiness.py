"""FE-032 regression: deterministic, Teacher-free Stage-2 (reference_validation) criterion-evidence
readiness for an evidence-bearing (``recovered-original-holdout``) reference.

Root cause these tests lock down: the authoritative source->split crosswalk manifest was declared in
both reference.yaml and workflow.yaml but never passed into ``build_split_crosswalk``, so every
recovered-holdout frame reported ``source_split_unjoined`` / domain "unknown" even though each frame
already carried the exact join keys. The readiness computation surfaces the deterministic lineage /
identity / split evidence BEFORE any Teacher/GPU dispatch (preflight) and INTO the gate packet.

Network- and Teacher-free; skips ASE/pydantic-dependent cases if those extras are absent.
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import yaml
    from ase import Atoms
    from ase.io import write
    _HAS_ASE = True
except ImportError:  # pragma: no cover
    _HAS_ASE = False

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


def _teacher_config(root: Path) -> Path:
    model = root / "teacher.nequip.pth"
    model.write_text("teacher-checkpoint-bytes")
    cfg = root / "teacher.yaml"
    cfg.write_text(yaml.safe_dump({"kind": "mock", "checkpoint": str(model)}))
    return cfg


def _frames(n=3, *, drop_labels_from=None):
    frames = []
    for i in range(n):
        a = Atoms("Si2", positions=[[0, 0, 0], [2.3 + i * 0.05, 0, 0]], cell=[12, 12, 12], pbc=True)
        a.info["source_category"] = "silicon_crystalline_main"
        a.info["source_local_index"] = i
        a.info["config_type"] = "bulk" if i % 2 == 0 else "surface"
        if drop_labels_from is None or i < drop_labels_from:
            a.info["dft_energy"] = -5.0 - i
            a.arrays["dft_forces"] = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
        frames.append(a)
    return frames


def _write_manifest(path: Path, *, categories_locals_splits):
    records = [
        {"global_index": gi, "source_category": cat, "source_local_index": loc, "split": split}
        for gi, (cat, loc, split) in enumerate(categories_locals_splits)
    ]
    path.write_text(json.dumps({"records": records}, indent=2))
    return path


def _build_package(root: Path, *, n=3, target_split="test", frame_count=None,
                   bad_structs_sha=False, drop_labels_from=None,
                   manifest_split_for=None):
    """Build an evidence-bearing recovered-original-holdout reference package.

    Returns ``(reference_yaml, structures_path, manifest_paths, teacher_config)``.
    ``manifest_split_for`` maps local_index -> list-of-(split) to inject an ambiguous key across
    multiple manifests; default assigns every frame ``target_split`` in a single manifest.
    """
    from workflow.integrity import sha256_file
    frames = _frames(n, drop_labels_from=drop_labels_from)
    structures = root / "recovered_original_holdout_test.xyz"
    write(str(structures), frames)

    if manifest_split_for is None:
        manifest = _write_manifest(
            root / "split_manifest.json",
            categories_locals_splits=[("silicon_crystalline_main", i, target_split) for i in range(n)])
        manifest_paths = [manifest]
    else:
        # one manifest per split alternative, producing a conflicting (ambiguous) key
        manifest_paths = []
        alternatives = {}
        for loc, splits in manifest_split_for.items():
            for k, s in enumerate(splits):
                alternatives.setdefault(k, []).append(("silicon_crystalline_main", loc, s))
        for k, recs in alternatives.items():
            manifest_paths.append(_write_manifest(root / f"split_manifest_{k}.json",
                                                  categories_locals_splits=recs))

    struct_sha = sha256_file(structures) if not bad_structs_sha else "0" * 64
    doc = {
        "kind": "recovered-original-holdout",
        "reference_id": "synthetic-recovered-holdout",
        "target_split": target_split,
        "frame_count": frame_count if frame_count is not None else n,
        "split_source_manifest": str(manifest_paths[0]),
        "split_source_manifest_sha256": sha256_file(manifest_paths[0]),
        "prohibited_uses": ["student_training", "acquisition_seed", "augmentation_parent"],
        "structures": {"path": str(structures), "logical_frames": n, "sha256": struct_sha},
    }
    reference = root / "reference.yaml"
    reference.write_text(yaml.safe_dump(doc))
    teacher = _teacher_config(root)
    return reference, structures, manifest_paths, teacher


def _controller(root: Path, structures: Path):
    return types.SimpleNamespace(state={
        "project_dir": str(root),
        "inputs": [{"source": str(structures)}],
        "protected_reference_roles": [],
    })


def _proposal(reference: Path, teacher: Path, **extra):
    params = {"reference_yaml": str(reference), "teacher_config": str(teacher)}
    params.update(extra)
    return {"action_type": "validate_teacher_reference", "parameters": params}


@unittest.skipUnless(_HAS_ASE, "ase/yaml/numpy not installed")
class ReferenceValidationReadinessTests(unittest.TestCase):
    def _compute(self, controller, proposal, manifest_paths, report_path=None):
        from runtimes.pydantic_ai.reference_validation_readiness import (
            compute_reference_validation_evidence)
        return compute_reference_validation_evidence(
            controller, proposal, split_manifest_paths=[str(p) for p in manifest_paths],
            report_path=report_path)

    # 1: complete lineage joined=all / unjoined=0 / correct split -> ready
    def test_complete_lineage_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, manifests, teacher = _build_package(root, n=3)
            out = self._compute(_controller(root, structures), _proposal(reference, teacher), manifests)
            self.assertIsNotNone(out)
            self.assertTrue(out["ready"], out["blocking_gaps"])
            self.assertEqual(out["blocking_gaps"], [])
            join = out["criteria"]["source_split_lineage_join"]
            self.assertEqual(join["status"], "COMPLETE")
            self.assertEqual(join["lineage_joined"], 3)
            self.assertEqual(join["lineage_unjoined"], 0)
            self.assertEqual(join["lineage_ambiguous"], 0)
            self.assertEqual(out["criteria"]["test_split_membership"]["status"], "CONFIRMED")
            self.assertEqual(join["split_distribution"], {"test": 3})

    # 2: missing lineage (no crosswalk manifest bound) -> fail closed
    def test_missing_lineage_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, _manifests, teacher = _build_package(root, n=3)
            out = self._compute(_controller(root, structures), _proposal(reference, teacher),
                                manifest_paths=[])  # authoritative manifest not surfaced
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            self.assertEqual(out["criteria"]["split_manifest_binding"]["status"], "UNBOUND")
            self.assertEqual(out["criteria"]["source_split_lineage_join"]["status"], "INCOMPLETE")
            self.assertTrue(any("crosswalk" in g or "lineage join" in g
                                for g in out["blocking_gaps"]))

    # 3: ambiguous lineage (same key -> two different splits) -> fail closed
    def test_ambiguous_lineage_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # local_index 0 appears as BOTH test and train across two manifests -> ambiguous key
            reference, structures, manifests, teacher = _build_package(
                root, n=1, manifest_split_for={0: ["test", "train"]})
            out = self._compute(_controller(root, structures), _proposal(reference, teacher), manifests)
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            join = out["criteria"]["source_split_lineage_join"]
            self.assertEqual(join["status"], "INCOMPLETE")
            self.assertEqual(join["lineage_ambiguous"], 1)

    # 4: bound-structure hash mismatch -> fail closed
    def test_structure_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, manifests, teacher = _build_package(root, n=3, bad_structs_sha=True)
            out = self._compute(_controller(root, structures), _proposal(reference, teacher), manifests)
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            self.assertEqual(out["criteria"]["structure_identity"]["status"], "MISMATCH")
            self.assertTrue(any("sha256 mismatch" in g for g in out["blocking_gaps"]))

    # 5: DFT-label provenance incomplete -> fail closed
    def test_dft_label_provenance_incomplete_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # only first frame keeps labels; frames 1,2 have no dft energy/forces
            reference, structures, manifests, teacher = _build_package(root, n=3, drop_labels_from=1)
            out = self._compute(_controller(root, structures), _proposal(reference, teacher), manifests)
            self.assertIsNotNone(out)
            self.assertFalse(out["ready"])
            dft = out["criteria"]["dft_label_provenance"]
            self.assertEqual(dft["status"], "INCOMPLETE")
            self.assertEqual(dft["frames_with_dft_energy"], 1)

    # 6: packet includes the deterministic provenance criteria (surfacing contract)
    def test_packet_includes_deterministic_provenance_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, manifests, teacher = _build_package(root, n=3)
            out = self._compute(_controller(root, structures), _proposal(reference, teacher), manifests)
            self.assertIsNotNone(out)
            expected = {
                "population_identity", "structure_identity", "split_manifest_binding",
                "source_split_lineage_join", "declared_frame_count_consistency",
                "test_split_membership", "dft_label_provenance", "teacher_checkpoint_identity",
                "no_historical_reuse", "protected_reference_use_policy",
                "prediction_artifact_identity", "global_fidelity_metrics", "grouped_fidelity_metrics",
            }
            self.assertTrue(expected <= set(out["criteria"]), expected - set(out["criteria"]))
            # numeric global fidelity carries units/denominators only after execution
            self.assertEqual(out["criteria"]["global_fidelity_metrics"]["status"], "PENDING_EXECUTION")

    # 7: the record is canonical/deterministic -> all 3 Judges receive the same packet
    def test_canonical_record_is_identical_across_computations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, manifests, teacher = _build_package(root, n=3)
            c = _controller(root, structures)
            p = _proposal(reference, teacher)
            first = self._compute(c, p, manifests)
            second = self._compute(c, p, manifests)
            self.assertEqual(json.dumps(first, sort_keys=True, default=str),
                             json.dumps(second, sort_keys=True, default=str))

    # 8: lineage readiness is established with NO Teacher execution
    def test_lineage_readiness_without_teacher_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, structures, manifests, teacher = _build_package(root, n=3)
            # report_path=None => pre-execution preflight; no reference_validation.json exists yet
            out = self._compute(_controller(root, structures), _proposal(reference, teacher),
                                manifests, report_path=None)
            self.assertIsNotNone(out)
            self.assertTrue(out["ready"])  # non-Teacher lineage/identity fully established
            self.assertTrue(out["teacher_metrics_pending"])
            for k in ("prediction_artifact_identity", "global_fidelity_metrics",
                      "grouped_fidelity_metrics"):
                self.assertEqual(out["criteria"][k]["status"], "PENDING_EXECUTION")

    def test_not_applicable_for_non_evidence_bearing_reference(self):
        # a non-evidence-bearing reference (no split_source_manifest) is out of scope: returns None
        # so the caller imposes no extra preflight/packet surfacing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.yaml"
            reference.write_text(yaml.safe_dump({"kind": "protected-existing-dft",
                                                 "reference_id": "x"}))
            teacher = _teacher_config(root)
            out = self._compute(_controller(root, reference), _proposal(reference, teacher), [])
            self.assertIsNone(out)


# 9: RootCauseValidationError accepts a typed evidence_gap classification whose affected_channel
# NEGATES the Teacher-vs-DFT channel in prose (the FE-032 secondary defect: naive substring match on
# "dft" auto-rejected a correctly-typed evidence_gap because "Teacher-vs-DFT" appeared inside a
# negated phrase).
@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class RootCauseNegatedDftChannelTests(unittest.TestCase):
    AVAILABLE = {"runs/r/eval/errorc.json", "runs/r/committee/u.json"}
    TARGETS = {"reference_validation", "teacher_baseline"}

    def _classification(self, **over):
        from runtimes.pydantic_ai.root_cause import RootCauseClassification
        from runtimes.pydantic_ai.models import EvidenceReference
        base = dict(
            run_id="r", stage="reference_validation", failure_category="evidence_gap",
            affected_channel="reference-validation evidence/provenance "
                             "(not a Teacher-vs-DFT accuracy-disagreement channel)",
            affected_artifact_refs=[EvidenceReference(role="simulation", path="runs/r/eval/errorc.json")],
            evidence_refs=[EvidenceReference(role="simulation", path="runs/r/eval/errorc.json"),
                           EvidenceReference(role="simulation", path="runs/r/committee/u.json")],
            evidence_summary="per-frame source->split lineage not surfaced in the bounded packet",
            confidence=0.8, excluded_alternatives=["reference_disagreement"],
            recommended_recovery_target="reference_validation",
            recommended_next_action="surface authoritative split crosswalk into the gate packet")
        base.update(over)
        return RootCauseClassification(**base)

    def _validate(self, c, **over):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        kwargs = dict(available_artifacts=self.AVAILABLE, valid_recovery_targets=self.TARGETS,
                      dft_comparison_evidence_present=False,
                      gate_alleges_accuracy_disagreement=False)
        kwargs.update(over)
        return validate_root_cause_classification(c, **kwargs)

    def test_typed_evidence_gap_with_negated_dft_prose_accepted(self):
        c = self._validate(self._classification())
        self.assertEqual(c.failure_category.value if hasattr(c.failure_category, "value")
                         else c.failure_category, "evidence_gap")

    def test_affirmative_dft_channel_still_rejected(self):
        # guard against over-widening: an AFFIRMATIVE Teacher-vs-DFT channel with no supporting
        # comparison evidence must still be rejected.
        from runtimes.pydantic_ai.root_cause import RootCauseValidationError
        c = self._classification(failure_category="reference_disagreement",
                                 affected_channel="teacher_vs_dft accuracy disagreement",
                                 recommended_recovery_target="teacher_baseline")
        with self.assertRaises(RootCauseValidationError):
            self._validate(c)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
