"""R26 forensic-audit regression tests (narrow fix, not a broader audit): two independent
defects in the SAME run.

1. ``ORIGINAL_HELDOUT_FIDELITY`` execution binding: R26 committed a ``teacher_validation_plan``
   selecting ``ORIGINAL_HELDOUT_FIDELITY``, but ``validate_teacher_reference`` executed against
   the historical ``protected-existing-dft`` reference instead of the recovered original
   held-out split -- because only the historical reference happened to be bound, it was silently
   substituted. See ``runtimes.pydantic_ai.cli._resolve_teacher_reference_binding``.

2. Recovery misdiagnosis from Gate semantics: R26 recovery 1 classified a ``reference_validation``
   REVISE as ``reference_disagreement`` merely because the stage's own evidence contains a
   Teacher-vs-DFT comparison (true of every reference_validation run), when every Judge's actual
   ``required_fix``/``rationale`` was about evidence-exposure/provenance/lineage-mapping
   completeness, not a demonstrated accuracy failure. See
   ``runtimes.pydantic_ai.cli._gate_alleges_accuracy_disagreement`` and its threading into
   ``root_cause.validate_root_cause_classification`` /
   ``recovery_bridge.validate_recovery_plan_proposal``.

Network-free; skips without the ``pydantic`` extra.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


# =============================================================================================
# Issue 1: ORIGINAL_HELDOUT_FIDELITY must execute against the recovered original held-out split,
# never against the historical protected-existing-dft reference, even when both are bound.
# =============================================================================================

def _historical_protected_reference(root: Path) -> Path:
    """A minimal, valid ``protected-existing-dft`` package (the historical population)."""
    from workflow.integrity import sha256_file
    a = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    structures = root / "historical_protected.xyz"
    write(str(structures), [a])
    indices = root / "historical_indices.txt"
    indices.write_text("760\n761\n")
    manifest = root / "historical_manifest.json"
    manifest.write_text(json.dumps({"mapping": {
        "logical_test_frames": 1, "matched_logical_frames": 1, "unmatched_logical_frames": 0,
        "protected_source_rows": 2, "conflicting_label_duplicates": 0,
    }}))
    reference = root / "historical_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "protected-existing-dft",
        "reference_id": "historical-protected-reference",
        "reference_class": "ORIGINAL_TEACHER_TEST",
        "status": "AVAILABLE_AND_PROTECTED",
        "logical_test_frames": 1,
        "protected_source_rows": 2,
        "protection_manifest": str(manifest),
        "protected_source_rows_file": str(indices),
        "duplicate_equivalent": {"source_global_indices": [760, 761], "label_conflict": False},
        "prohibited_uses": ["student_training", "student_validation_tuning", "acquisition_seed",
                            "augmentation_parent", "recovery_training"],
        "structures": {"path": str(structures), "logical_frames": 1, "sha256": sha256_file(structures)},
    }))
    return reference


def _recovered_holdout_reference(root: Path, *, target_split="test") -> Path:
    """A minimal, valid ``recovered-original-holdout`` package (the genuine held-out split),
    distinct in content and provenance from the historical reference above."""
    from workflow.integrity import sha256_file
    from validation.protected_reference import RECOVERED_HOLDOUT_REFERENCE_CLASS

    split_records = [
        {"source_category": "bulk", "source_local_index": 0, "split": "train"},
        {"source_category": "bulk", "source_local_index": 1, "split": target_split},
        {"source_category": "bulk", "source_local_index": 2, "split": target_split},
    ]
    manifest = root / "recovered_split_manifest.json"
    manifest.write_text(json.dumps({"records": split_records}))

    def _frame(local_index, x):
        atoms = Atoms("Cu", positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.info["source_category"] = "bulk"
        atoms.info["source_local_index"] = local_index
        atoms.info["dft_energy"] = -1.0
        atoms.arrays["dft_forces"] = np.array([[0.0, 0.0, 0.0]])
        return atoms

    frames = [_frame(1, 1.0), _frame(2, 2.0)]
    structures = root / "recovered_holdout.xyz"
    write(str(structures), frames)

    reference = root / "recovered_reference.yaml"
    reference.write_text(yaml.safe_dump({
        "kind": "recovered-original-holdout",
        "reference_id": "recovered-original-holdout-" + target_split,
        "reference_class": RECOVERED_HOLDOUT_REFERENCE_CLASS,
        "status": "AVAILABLE_AND_VERIFIED",
        "target_split": target_split,
        "split_source_manifest": str(manifest),
        "split_source_manifest_sha256": sha256_file(manifest),
        "frame_count": len(frames),
        "structures": {"path": str(structures), "logical_frames": len(frames),
                      "sha256": sha256_file(structures)},
        "prohibited_uses": ["student_training", "student_validation_tuning", "acquisition_seed",
                            "augmentation_parent", "recovery_training"],
    }))
    return reference


def _teacher_config(root: Path) -> Path:
    model = root / "teacher.nequip.pth"
    model.write_text("teacher")
    cfg = root / "teacher.yaml"
    cfg.write_text(yaml.safe_dump({"kind": "mock", "checkpoint": str(model)}))
    return cfg


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class HeldoutFidelityExecutionBindingTests(unittest.TestCase):
    """Reproduces R26 exactly, then proves the fix: a committed plan selecting
    ORIGINAL_HELDOUT_FIDELITY with both a recovered-holdout AND a historical protected reference
    bound must resolve to the recovered reference -- the historical one is never substituted."""

    def _init_controller(self, root, *, teacher, historical_reference, recovered_reference):
        from workflow.controller import RunController
        workflow = root / "workflow.yaml"
        workflow.write_text(yaml.safe_dump({
            "run_id": "r26-heldout-binding-fix",
            "inputs": [str(teacher), str(historical_reference), str(recovered_reference)],
            "stages": [{
                "name": "reference_validation",
                "command": None,
                "outputs": ["artifacts/reference_validation.json",
                           "artifacts/teacher_reference_predictions.extxyz"],
                "gate": {"criteria": ["reference validation report is valid"]},
            }],
        }))
        return RunController.initialize(workflow, root / "run")

    def test_original_heldout_fidelity_binds_recovered_not_historical(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = _teacher_config(root)
            historical_reference = _historical_protected_reference(root)
            recovered_reference = _recovered_holdout_reference(root, target_split="test")
            c = self._init_controller(root, teacher=teacher, historical_reference=historical_reference,
                                      recovered_reference=recovered_reference)
            # The run's committed teacher_validation_plan selects ORIGINAL_HELDOUT_FIDELITY,
            # exactly as R26's did (evidence_profile.resolved_heldout_split: "test").
            c.state["teacher_validation_plan"] = {
                "selected_components": ["ORIGINAL_HELDOUT_FIDELITY"], "target_split": "test",
            }
            proposal, role = _proposal_from_stage(c, "reference_validation",
                                                  _stage_config(c, "reference_validation"))
            self.assertEqual(role, "simulation")
            bound_reference_yaml = proposal["parameters"]["reference_yaml"]
            recovered_snapshot = str(Path(c.state["inputs"][2]["snapshot"]).resolve())
            historical_snapshot = str(Path(c.state["inputs"][1]["snapshot"]).resolve())
            self.assertEqual(bound_reference_yaml, recovered_snapshot)
            self.assertNotEqual(bound_reference_yaml, historical_snapshot)

    def test_target_split_mismatch_between_plan_and_bound_reference_fails_closed(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = _teacher_config(root)
            historical_reference = _historical_protected_reference(root)
            # Recovered reference declares "test", but the committed plan asks for "val".
            recovered_reference = _recovered_holdout_reference(root, target_split="test")
            c = self._init_controller(root, teacher=teacher, historical_reference=historical_reference,
                                      recovered_reference=recovered_reference)
            c.state["teacher_validation_plan"] = {
                "selected_components": ["ORIGINAL_HELDOUT_FIDELITY"], "target_split": "val",
            }
            with self.assertRaises(ValueError) as ctx:
                _proposal_from_stage(c, "reference_validation", _stage_config(c, "reference_validation"))
            self.assertIn("target_split", str(ctx.exception))

    def test_missing_recovered_reference_fails_closed_never_falls_back_to_historical(self):
        from runtimes.pydantic_ai.cli import _proposal_from_stage, _stage_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = _teacher_config(root)
            historical_reference = _historical_protected_reference(root)
            # Bind only the historical reference; the run's inputs list must still validate, so
            # reuse it twice as a harmless placeholder for the required workflow.yaml shape.
            from workflow.controller import RunController
            workflow = root / "workflow.yaml"
            workflow.write_text(yaml.safe_dump({
                "run_id": "r26-heldout-binding-missing",
                "inputs": [str(teacher), str(historical_reference)],
                "stages": [{
                    "name": "reference_validation", "command": None,
                    "outputs": ["artifacts/reference_validation.json",
                               "artifacts/teacher_reference_predictions.extxyz"],
                    "gate": {"criteria": ["reference validation report is valid"]},
                }],
            }))
            c = RunController.initialize(workflow, root / "run")
            c.state["teacher_validation_plan"] = {
                "selected_components": ["ORIGINAL_HELDOUT_FIDELITY"], "target_split": "test",
            }
            with self.assertRaises(ValueError) as ctx:
                _proposal_from_stage(c, "reference_validation", _stage_config(c, "reference_validation"))
            self.assertIn("recovered-original-holdout", str(ctx.exception))


# =============================================================================================
# Issue 2: a Gate REVISE grounded purely in evidence-exposure/provenance/lineage-mapping language
# must never be classified as a Teacher-vs-DFT accuracy disagreement, and must never by itself
# authorize fresh DFT / teacher relabeling / Student retraining.
# =============================================================================================

def _provenance_only_vote_bundle():
    """Every non-PASS judge's rationale/required_fix is exclusively about exposing existing
    counts/manifests/lineage mappings -- exactly R26's actual judge text (see
    runs/sio2-sox-allegro-simplenn-r26/exchange/results/reference_validation-judge-{1,2}.json):
    no judge alleges the underlying Teacher-vs-DFT comparison itself disagreed."""
    return {"votes": [
        {"judge_id": "judge-1", "review_lens": "provenance", "verdict": "REVISE",
         "rationale": "The mapping manifest does not expose zero unmatched lineage frames inline.",
         "required_fix": "Expose the source-pool crosswalk and lineage mapping counts in the report."},
        {"judge_id": "judge-2", "review_lens": "evidence_completeness", "verdict": "REVISE",
         "rationale": "Per-domain metric tables and the protected-use policy text are not visible "
                      "in the packet, only referenced.",
         "required_fix": "Include the explicit policy excerpt and per-config-type metric tables."},
        {"judge_id": "judge-3", "review_lens": "accuracy", "verdict": "PASS",
         "rationale": "Deterministic validation passed; Teacher-vs-DFT metrics were computed for "
                      "all held-out frames.",
         "required_fix": ""},
    ]}


def _accuracy_disagreement_vote_bundle():
    """Contrast fixture: a judge's own text DOES allege a genuine accuracy problem."""
    bundle = _provenance_only_vote_bundle()
    bundle["votes"][0]["rationale"] = "Teacher-vs-DFT force error shows a systematic bias on this split."
    bundle["votes"][0]["required_fix"] = "Investigate the Teacher/DFT disagreement and consider relabeling."
    return bundle


AVAILABLE_ARTIFACTS = {"runs/r26/reference_validation.json"}
VALID_RECOVERY_TARGETS = {"reference_validation", "acquire_structures"}
CAPABILITY_ROSTER = {"teacher_relabel_capability": "data-curator"}
VALID_STAGE_NAMES = {"reference_validation", "acquire_structures"}


def _reference_disagreement_classification():
    from runtimes.pydantic_ai.root_cause import RootCauseClassification
    from runtimes.pydantic_ai.models import EvidenceReference
    return RootCauseClassification(
        run_id="r26", stage="reference_validation", failure_category="reference_disagreement",
        affected_channel="teacher_vs_dft",
        evidence_refs=[EvidenceReference(role="reference_validation",
                                        path="runs/r26/reference_validation.json")],
        evidence_summary="Context indicates DFT-comparison evidence is present in the stage artifacts.",
        confidence=0.7, recommended_recovery_target="reference_validation",
        recommended_next_action="acquire new DFT labels and retrain the teacher")


def _relabel_and_retrain_proposal():
    from runtimes.pydantic_ai.recovery_bridge import RecoveryPlanProposal
    return RecoveryPlanProposal(
        run_id="r26", failed_stage="reference_validation", diagnosis_artifact_sha256="d" * 64,
        capability="teacher_relabel_capability", return_stage="reference_validation",
        proposed_changes=[{"type": "relabel_and_retrain"}],
        labeling={"teacher_relabel": True, "new_dft": True},
        student_training={"retrain": True, "mode": "full"},
        revalidation={"reuse_profile": False, "targets": ["reference_validation"]},
        rationale="teacher disagrees with DFT on this split")


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class GateGroundedRecoveryDiagnosisTests(unittest.TestCase):
    def test_gate_alleges_accuracy_disagreement_is_false_for_provenance_only_revise(self):
        from runtimes.pydantic_ai.cli import _gate_alleges_accuracy_disagreement
        self.assertFalse(_gate_alleges_accuracy_disagreement(_provenance_only_vote_bundle()))

    def test_gate_alleges_accuracy_disagreement_is_true_when_a_judge_actually_says_so(self):
        from runtimes.pydantic_ai.cli import _gate_alleges_accuracy_disagreement
        self.assertTrue(_gate_alleges_accuracy_disagreement(_accuracy_disagreement_vote_bundle()))

    def test_pending_gate_vote_bundle_matches_stage_and_timestamp_not_stage_alone(self):
        from runtimes.pydantic_ai.cli import _pending_gate_vote_bundle
        controller = type("C", (), {"state": {"events": [
            {"type": "gate", "stage": "reference_validation", "at": "t1",
             "vote_bundle": _accuracy_disagreement_vote_bundle()},
            {"type": "gate", "stage": "reference_validation", "at": "t2",
             "vote_bundle": _provenance_only_vote_bundle()},
        ]}})()
        pending = {"failed_stage": "reference_validation", "gate_recorded_at": "t2"}
        self.assertEqual(_pending_gate_vote_bundle(controller, pending),
                         _provenance_only_vote_bundle())

    def test_reference_disagreement_classification_rejected_without_gate_grounding(self):
        from runtimes.pydantic_ai.root_cause import (
            RootCauseValidationError, validate_root_cause_classification)
        c = _reference_disagreement_classification()
        with self.assertRaises(RootCauseValidationError):
            validate_root_cause_classification(
                c, available_artifacts=AVAILABLE_ARTIFACTS,
                valid_recovery_targets=VALID_RECOVERY_TARGETS,
                dft_comparison_evidence_present=True,  # the stage DOES contain a DFT comparison
                gate_alleges_accuracy_disagreement=False)  # but no judge alleged a disagreement

    def test_reference_disagreement_classification_accepted_when_gate_actually_alleges_it(self):
        from runtimes.pydantic_ai.root_cause import validate_root_cause_classification
        c = _reference_disagreement_classification()
        validated = validate_root_cause_classification(
            c, available_artifacts=AVAILABLE_ARTIFACTS, valid_recovery_targets=VALID_RECOVERY_TARGETS,
            dft_comparison_evidence_present=True, gate_alleges_accuracy_disagreement=True)
        self.assertEqual(validated.failure_category, "reference_disagreement")

    def test_new_dft_and_retrain_proposal_rejected_without_gate_grounding(self):
        from runtimes.pydantic_ai.recovery_bridge import (
            RecoveryPlanValidationError, validate_recovery_plan_proposal)
        p = _relabel_and_retrain_proposal()
        with self.assertRaises(RecoveryPlanValidationError) as ctx:
            validate_recovery_plan_proposal(
                p, expected_failed_stage="reference_validation", expected_diagnosis_sha256="d" * 64,
                capability_roster=CAPABILITY_ROSTER, valid_stage_names=VALID_STAGE_NAMES,
                dft_comparison_evidence_present=True, gate_alleges_accuracy_disagreement=False)
        message = str(ctx.exception)
        self.assertIn("labeling.new_dft", message)
        self.assertIn("labeling.teacher_relabel", message)
        self.assertIn("student_training.retrain", message)

    def test_new_dft_and_retrain_proposal_accepted_when_gate_actually_alleges_disagreement(self):
        from runtimes.pydantic_ai.recovery_bridge import validate_recovery_plan_proposal
        p = _relabel_and_retrain_proposal()
        validated = validate_recovery_plan_proposal(
            p, expected_failed_stage="reference_validation", expected_diagnosis_sha256="d" * 64,
            capability_roster=CAPABILITY_ROSTER, valid_stage_names=VALID_STAGE_NAMES,
            dft_comparison_evidence_present=True, gate_alleges_accuracy_disagreement=True)
        self.assertTrue(validated.labeling.new_dft)

    def test_evidence_gathering_only_proposal_unaffected_by_gate_grounding(self):
        # A proposal that authorizes NONE of the three gated actions must not be rejected merely
        # because no judge alleged an accuracy disagreement -- only the three named actions are
        # gated by this check.
        from runtimes.pydantic_ai.recovery_bridge import (
            RecoveryPlanProposal, validate_recovery_plan_proposal)
        p = RecoveryPlanProposal(
            run_id="r26", failed_stage="reference_validation", diagnosis_artifact_sha256="d" * 64,
            capability="teacher_relabel_capability", return_stage="reference_validation",
            proposed_changes=[{"type": "expose_lineage_mapping"}],
            labeling={"teacher_relabel": False, "new_dft": False},
            student_training={"retrain": False, "mode": "none"},
            revalidation={"reuse_profile": True, "targets": ["reference_validation"]},
            rationale="expose the existing lineage mapping and metric tables the judges asked for")
        validated = validate_recovery_plan_proposal(
            p, expected_failed_stage="reference_validation", expected_diagnosis_sha256="d" * 64,
            capability_roster=CAPABILITY_ROSTER, valid_stage_names=VALID_STAGE_NAMES,
            dft_comparison_evidence_present=True, gate_alleges_accuracy_disagreement=False)
        self.assertFalse(validated.labeling.new_dft)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
