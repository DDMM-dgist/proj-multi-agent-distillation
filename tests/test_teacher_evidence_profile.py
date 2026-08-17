"""Tests for the generic, evidence-driven Teacher validation ADMISSIBLE COMPONENT MODEL
(validation.teacher_evidence_profile). Every scenario here uses only boolean evidence facts --
never a material, dataset, or campaign name -- proving the mechanism is a reference
demonstration of a generic, evidence-driven decision, not a SiO2/Allegro-specific branch.

Components are additive (not a mutually-exclusive strategy enum): a profile's admissible set
may, and often does, contain more than one member.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from validation.teacher_evidence_profile import (
    CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING,
    DEPLOYMENT_APPLICABILITY,
    INDEPENDENT_REFERENCE_FIDELITY,
    OPERATIONAL_ROBUSTNESS,
    ORIGINAL_HELDOUT_FIDELITY,
    PROTECTED_DATA_RESTRICTIONS,
    SPLIT_ROLE_HELDOUT_EVALUATION,
    SPLIT_ROLE_TRAINING,
    SPLIT_ROLE_VALIDATION,
    TRAINING_CORPUS_CONSISTENCY,
    TeacherEvidenceProfile,
    _frame_has_finite_labels,
    _load_positional_split_manifest,
    _load_split_roles,
    _merged_split_roles,
    _resolve_unique_heldout_role_split,
    _verified_positional_split_join,
    derive_admissible_decision_space,
    inspect_teacher_evidence,
)


def _profile(**overrides):
    base = dict(
        teacher_model_available=True,
        operational_evaluation_population_available=False,
        original_training_db_available=False,
        original_labels_available=False,
        original_split_recovered=False,
        genuine_holdout_test_available=False,
        independent_external_reference_available=False,
        deployment_domain_population_available=False,
    )
    base.update(overrides)
    return TeacherEvidenceProfile(**base)


class AdmissibleDecisionSpaceTests(unittest.TestCase):
    def test_no_evidence_at_all_is_the_insufficient_evidence_floor(self):
        result = derive_admissible_decision_space(_profile())
        self.assertEqual(result["admissible_components"], [])
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["floor"], CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING)

    def test_custom_teacher_with_trustworthy_original_holdout(self):
        # custom Teacher + DB + labels + trustworthy split + genuine holdout + an operational
        # population -> {OPERATIONAL_ROBUSTNESS, ORIGINAL_HELDOUT_FIDELITY}. Note
        # TRAINING_CORPUS_CONSISTENCY's weaker requirement is trivially also satisfied here, so
        # it must be admissible too -- components are additive, never mutually exclusive.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY,
                          ORIGINAL_HELDOUT_FIDELITY})
        self.assertIn("held_out_fidelity",
                      result["components"][ORIGINAL_HELDOUT_FIDELITY]["allowed_claims"])

    def test_custom_teacher_with_db_but_no_trustworthy_split(self):
        # custom Teacher + DB but no trustworthy split -> {OPERATIONAL_ROBUSTNESS,
        # TRAINING_CORPUS_CONSISTENCY}; original held-out fidelity is not admissible.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY})
        self.assertNotIn(ORIGINAL_HELDOUT_FIDELITY, result["admissible_components"])
        self.assertNotIn("held_out_fidelity",
                         result["components"][TRAINING_CORPUS_CONSISTENCY]["allowed_claims"])
        self.assertIn("held_out_fidelity",
                      result["components"][TRAINING_CORPUS_CONSISTENCY]["prohibited_claims"])

    def test_umlip_with_genuine_independent_external_reference(self):
        # uMLIP (no DB) + genuine independent external reference + an operational population
        # -> {OPERATIONAL_ROBUSTNESS, INDEPENDENT_REFERENCE_FIDELITY}.
        profile = _profile(
            operational_evaluation_population_available=True,
            independent_external_reference_available=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, INDEPENDENT_REFERENCE_FIDELITY})

    def test_umlip_with_no_reference_but_an_operational_population_is_not_empty(self):
        # uMLIP + no independent reference, but the Teacher model plus a real operational
        # evaluation population exist -> {OPERATIONAL_ROBUSTNESS} only. NOT the insufficient-
        # evidence floor: an operationally-evaluable Teacher is always plannable.
        profile = _profile(operational_evaluation_population_available=True)
        result = derive_admissible_decision_space(profile)
        self.assertEqual(result["admissible_components"], [OPERATIONAL_ROBUSTNESS])
        self.assertFalse(result["insufficient_evidence"])

    def test_teacher_model_alone_with_no_population_is_insufficient(self):
        # teacher_model_available alone, with no operational population, does NOT satisfy
        # OPERATIONAL_ROBUSTNESS -- a Teacher with nothing to evaluate against has no
        # operational-robustness evidence merely because it is loadable.
        profile = _profile(teacher_model_available=True)
        result = derive_admissible_decision_space(profile)
        self.assertEqual(result["admissible_components"], [])
        self.assertTrue(result["insufficient_evidence"])

    def test_original_holdout_plus_deployment_domain_mismatch(self):
        # original holdout + a deployment-domain mismatch -> ORIGINAL_HELDOUT_FIDELITY and
        # DEPLOYMENT_APPLICABILITY are distinct and both admissible; held-out fidelity alone
        # does not satisfy the deployment-applicability requirement.
        profile = _profile(
            operational_evaluation_population_available=True,
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=False,
        )
        result = derive_admissible_decision_space(profile)
        self.assertEqual(set(result["admissible_components"]),
                         {OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY,
                          ORIGINAL_HELDOUT_FIDELITY, DEPLOYMENT_APPLICABILITY})

    def test_matching_deployment_domain_does_not_admit_deployment_applicability(self):
        profile = _profile(
            original_training_db_available=True, original_labels_available=True,
            original_split_recovered=True, genuine_holdout_test_available=True,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=True,
        )
        result = derive_admissible_decision_space(profile)
        self.assertNotIn(DEPLOYMENT_APPLICABILITY, result["admissible_components"])

    def test_protected_data_restrictions_are_unconditional(self):
        # Present regardless of which components are admissible, including the empty case.
        for profile in (_profile(), _profile(independent_external_reference_available=True)):
            result = derive_admissible_decision_space(profile)
            self.assertEqual(set(result["protected_data_restrictions"]),
                             set(PROTECTED_DATA_RESTRICTIONS))

    def test_reference_sio2_allegro_campaign_evidence_resolves_as_expected(self):
        # This is the reference demonstration the user described: for THIS campaign, DB is
        # available, split is recovered and cross-version verified, and a genuine 1,142-frame
        # held-out test exists -- so the generic mechanism naturally admits
        # ORIGINAL_HELDOUT_FIDELITY (plus the weaker components its evidence subsumes). Nothing
        # here is a special-cased branch; it is the same decision logic exercised with this
        # campaign's actual evidence values.
        profile = TeacherEvidenceProfile(
            teacher_model_available=True,
            operational_evaluation_population_available=True,
            original_training_db_available=True,
            original_labels_available=True,
            original_split_recovered=True,
            original_split_confidence="VERIFIED_ZERO_DISCREPANCY_CROSS_VERSION",
            genuine_holdout_test_available=True,
            genuine_holdout_test_frame_count=1142,
            independent_external_reference_available=False,
            deployment_domain_population_available=True,
            deployment_domain_matches_original_test_distribution=None,
        )
        result = derive_admissible_decision_space(profile)
        self.assertIn(ORIGINAL_HELDOUT_FIDELITY, result["admissible_components"])
        self.assertNotIn(DEPLOYMENT_APPLICABILITY, result["admissible_components"])
        self.assertNotIn(INDEPENDENT_REFERENCE_FIDELITY, result["admissible_components"])


def _custom_key_frame(local_index, *, category="bulk", energy=-1.0, forces=None):
    a = Atoms("Cu", positions=[[local_index, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["source_category"] = category
    a.info["source_local_index"] = local_index
    a.info["dft_energy"] = energy
    a.new_array("dft_forces", forces if forces is not None else np.zeros((1, 3)))
    return a


def _ase_calc_frame(local_index, *, category="bulk", energy=-1.0, forces=None, calc=True):
    a = Atoms("Cu", positions=[[local_index, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["source_category"] = category
    a.info["source_local_index"] = local_index
    if calc:
        a.calc = SinglePointCalculator(
            a, energy=energy, forces=forces if forces is not None else np.zeros((1, 3)))
    return a


class ASELabelFallbackTests(unittest.TestCase):
    """Item 1: standard ASE calculator-backed labels, generically recognized -- only as a
    fallback when the configured custom-key convention is genuinely absent."""

    def test_custom_key_labels_still_recognized_unchanged(self):
        frame = _custom_key_frame(0)
        self.assertTrue(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_standard_ase_calculator_energy_forces_recognized(self):
        frame = _ase_calc_frame(0)
        self.assertIsNone(frame.info.get("dft_energy"))
        self.assertTrue(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_missing_calculator_fails_closed(self):
        frame = _ase_calc_frame(0, calc=False)
        self.assertFalse(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_non_finite_energy_fails_closed(self):
        frame = _ase_calc_frame(0, energy=float("nan"))
        self.assertFalse(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_non_finite_forces_fail_closed(self):
        frame = _ase_calc_frame(0, forces=np.array([[float("inf"), 0.0, 0.0]]))
        self.assertFalse(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_malformed_force_shape_fails_closed(self):
        frame = _ase_calc_frame(0, forces=np.zeros((2, 3)))  # 1 atom, but 2 force rows
        self.assertFalse(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))

    def test_malformed_custom_labels_do_not_fall_through_to_calculator(self):
        # A frame that DECLARES a custom-key label but stores it malformed must fail closed on
        # that path -- it must never silently fall through to a (possibly valid) calculator label.
        frame = _custom_key_frame(0, energy=float("nan"))
        frame.calc = SinglePointCalculator(frame, energy=-1.0, forces=np.zeros((1, 3)))
        self.assertFalse(
            _frame_has_finite_labels(frame, energy_key="dft_energy", forces_key="dft_forces"))


def _write_positional_manifest(path: Path, *, records, source_dataset_sha256, total_frames):
    payload = {
        "index_semantics": "source_dataset_positional_index",
        "source_dataset_sha256": source_dataset_sha256,
        "total_frames": total_frames,
        "records": records,
    }
    path.write_text(json.dumps(payload))


class PositionalSplitJoinTests(unittest.TestCase):
    """Item 2: the second, opt-in (source_dataset_sha256, positional_index) join representation
    -- admissible only after fail-closed digest/count/coverage verification against the REAL
    training-DB file."""

    def _write_db(self, root: Path, n=4):
        frames = [_ase_calc_frame(i) for i in range(n)]
        db_path = root / "db.extxyz"
        write(str(db_path), frames)
        import hashlib
        sha256 = hashlib.sha256(db_path.read_bytes()).hexdigest()
        return db_path, sha256

    def test_positional_join_succeeds_with_matching_digest_count_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            db_frames = [_ase_calc_frame(i) for i in range(4)]
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                records=[{"global_index": i, "split": "train" if i < 3 else "test"}
                         for i in range(4)],
                source_dataset_sha256=sha256, total_frames=4)
            manifest_data = _load_positional_split_manifest(manifest_path)
            self.assertIsNotNone(manifest_data)
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=db_path, db_frames=db_frames)
            self.assertEqual(verified, {0: "train", 1: "train", 2: "train", 3: "test"})

    def test_digest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _real_sha256 = self._write_db(root, n=4)
            db_frames = [_ase_calc_frame(i) for i in range(4)]
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                records=[{"global_index": i, "split": "train"} for i in range(4)],
                source_dataset_sha256="0" * 64, total_frames=4)
            manifest_data = _load_positional_split_manifest(manifest_path)
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=db_path, db_frames=db_frames)
            self.assertIsNone(verified)

    def test_frame_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            db_frames = [_ase_calc_frame(i) for i in range(4)]
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                records=[{"global_index": i, "split": "train"} for i in range(4)],
                source_dataset_sha256=sha256, total_frames=5)  # declared count disagrees
            manifest_data = _load_positional_split_manifest(manifest_path)
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=db_path, db_frames=db_frames)
            self.assertIsNone(verified)

    def test_duplicate_positional_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                records=[{"global_index": 0, "split": "train"},
                         {"global_index": 0, "split": "test"},
                         {"global_index": 1, "split": "train"}],
                source_dataset_sha256="ab" * 32, total_frames=2)
            manifest_data = _load_positional_split_manifest(manifest_path)
            self.assertIsNone(manifest_data)

    def test_missing_positional_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            db_frames = [_ase_calc_frame(i) for i in range(4)]
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                # index 2 omitted entirely -- only 3 records for a declared 4-frame total.
                records=[{"global_index": i, "split": "train"} for i in (0, 1, 3)],
                source_dataset_sha256=sha256, total_frames=4)
            manifest_data = _load_positional_split_manifest(manifest_path)
            self.assertIsNotNone(manifest_data)  # loads fine; the omission is a coverage gap
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=db_path, db_frames=db_frames)
            self.assertIsNone(verified)  # record count (3) != declared total_frames (4): fail closed

    def test_out_of_range_positional_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            db_frames = [_ase_calc_frame(i) for i in range(4)]
            manifest_path = root / "manifest.json"
            _write_positional_manifest(
                manifest_path,
                records=[{"global_index": i, "split": "train"} for i in (0, 1, 2, 99)],
                source_dataset_sha256=sha256, total_frames=4)
            manifest_data = _load_positional_split_manifest(manifest_path)
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=db_path, db_frames=db_frames)
            self.assertIsNone(verified)


class InspectTeacherEvidenceJoinIntegrationTests(unittest.TestCase):
    """Item 2/3 integration: ``inspect_teacher_evidence`` end-to-end over both join
    representations, including its fail-closed conflict policy."""

    def _write_db(self, root: Path, n=4):
        frames = [_ase_calc_frame(i) for i in range(n)]
        db_path = root / "db.extxyz"
        write(str(db_path), frames)
        import hashlib
        sha256 = hashlib.sha256(db_path.read_bytes()).hexdigest()
        return db_path, sha256

    def test_category_local_index_join_still_works_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"records": [
                {"source_category": "bulk", "source_local_index": i,
                 "split": "train" if i < 3 else "test"} for i in range(4)]}))
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path],
                target_split="test")
            self.assertTrue(profile.original_split_recovered)
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.genuine_holdout_test_frame_count, 1)

    def test_both_join_representations_agreeing_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            category_manifest = root / "category_manifest.json"
            category_manifest.write_text(json.dumps({"records": [
                {"source_category": "bulk", "source_local_index": i,
                 "split": "train" if i < 3 else "test"} for i in range(4)]}))
            positional_manifest = root / "positional_manifest.json"
            _write_positional_manifest(
                positional_manifest,
                records=[{"global_index": i, "split": "train" if i < 3 else "test"}
                         for i in range(4)],
                source_dataset_sha256=sha256, total_frames=4)
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[category_manifest, positional_manifest],
                target_split="test")
            self.assertTrue(profile.original_split_recovered)
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.genuine_holdout_test_frame_count, 1)

    def test_both_join_representations_disagreeing_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, sha256 = self._write_db(root, n=4)
            # Category join says index 3 is "test"; positional join says index 3 is "train" --
            # a genuine cross-representation conflict.
            category_manifest = root / "category_manifest.json"
            category_manifest.write_text(json.dumps({"records": [
                {"source_category": "bulk", "source_local_index": i,
                 "split": "train" if i < 3 else "test"} for i in range(4)]}))
            positional_manifest = root / "positional_manifest.json"
            _write_positional_manifest(
                positional_manifest,
                records=[{"global_index": i, "split": "train"} for i in range(4)],
                source_dataset_sha256=sha256, total_frames=4)
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[category_manifest, positional_manifest],
                target_split="test")
            self.assertFalse(profile.original_split_recovered)
            self.assertFalse(profile.genuine_holdout_test_available)
            self.assertIsNone(profile.genuine_holdout_test_frame_count)


class SplitRoleProvenanceTests(unittest.TestCase):
    """Issue 1: a genuine held-out split is discoverable from PROVENANCE ALONE (a generic
    ``split_roles: {<split name>: training|validation|heldout_evaluation}`` declaration), never
    requiring a caller to pre-supply the literal split name -- and the actual name need not be
    "test"."""

    def _write_db(self, root: Path, n=4):
        frames = [_ase_calc_frame(i) for i in range(n)]
        db_path = root / "db.extxyz"
        write(str(db_path), frames)
        import hashlib
        sha256 = hashlib.sha256(db_path.read_bytes()).hexdigest()
        return db_path, sha256

    def _manifest(self, root, *, split_names, roles=None, name="manifest.json"):
        records = [{"source_category": "bulk", "source_local_index": i, "split": split_names[i]}
                   for i in range(len(split_names))]
        payload = {"records": records}
        if roles is not None:
            payload["split_roles"] = roles
        path = root / name
        path.write_text(json.dumps(payload))
        return path

    def test_load_split_roles_accepts_a_well_formed_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._manifest(root, split_names=["train"] * 3 + ["holdout_eval"],
                                  roles={"train": SPLIT_ROLE_TRAINING,
                                         "holdout_eval": SPLIT_ROLE_HELDOUT_EVALUATION})
            self.assertEqual(_load_split_roles(path),
                             {"train": SPLIT_ROLE_TRAINING,
                              "holdout_eval": SPLIT_ROLE_HELDOUT_EVALUATION})

    def test_load_split_roles_rejects_an_unknown_role_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._manifest(root, split_names=["train"],
                                  roles={"train": "not_a_real_role"})
            self.assertIsNone(_load_split_roles(path))

    def test_merged_split_roles_fails_closed_on_cross_manifest_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = self._manifest(root, split_names=["train", "test"],
                                roles={"test": SPLIT_ROLE_HELDOUT_EVALUATION}, name="m1.json")
            m2 = self._manifest(root, split_names=["train", "test"],
                                roles={"test": SPLIT_ROLE_VALIDATION}, name="m2.json")
            self.assertIsNone(_merged_split_roles([m1, m2]))

    def test_resolve_unique_heldout_role_split_fails_closed_on_zero_matches(self):
        merged = {"train": SPLIT_ROLE_TRAINING, "val": SPLIT_ROLE_VALIDATION}
        self.assertIsNone(_resolve_unique_heldout_role_split(merged, ["train", "val"]))

    def test_resolve_unique_heldout_role_split_fails_closed_on_multiple_matches(self):
        merged = {"holdA": SPLIT_ROLE_HELDOUT_EVALUATION, "holdB": SPLIT_ROLE_HELDOUT_EVALUATION}
        self.assertIsNone(
            _resolve_unique_heldout_role_split(merged, ["holdA", "holdB", "train"]))

    def test_resolve_unique_heldout_role_split_succeeds_with_a_non_test_name(self):
        # The genuine held-out split need not be named "test" -- any single name uniquely
        # carrying the heldout_evaluation role resolves.
        merged = {"train": SPLIT_ROLE_TRAINING, "holdout_eval": SPLIT_ROLE_HELDOUT_EVALUATION}
        self.assertEqual(
            _resolve_unique_heldout_role_split(merged, ["train", "holdout_eval"]),
            "holdout_eval")

    def test_inspect_teacher_evidence_admits_holdout_from_provenance_alone(self):
        # No target_split is ever passed -- ORIGINAL_HELDOUT_FIDELITY's evidence
        # (genuine_holdout_test_available) must still become available, from split_roles alone,
        # under a split name that is deliberately NOT "test".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = self._manifest(
                root, split_names=["train", "train", "train", "holdout_eval"],
                roles={"train": SPLIT_ROLE_TRAINING,
                       "holdout_eval": SPLIT_ROLE_HELDOUT_EVALUATION})
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path])
            self.assertTrue(profile.original_split_recovered)
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.genuine_holdout_test_frame_count, 1)
            self.assertEqual(profile.resolved_heldout_split, "holdout_eval")

    def test_inspect_teacher_evidence_without_any_heldout_role_is_inadmissible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = self._manifest(
                root, split_names=["train", "train", "train", "val"],
                roles={"train": SPLIT_ROLE_TRAINING, "val": SPLIT_ROLE_VALIDATION})
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path])
            self.assertTrue(profile.original_split_recovered)
            self.assertFalse(profile.genuine_holdout_test_available)
            self.assertIsNone(profile.genuine_holdout_test_frame_count)
            self.assertIsNone(profile.resolved_heldout_split)

    def test_inspect_teacher_evidence_with_no_split_roles_declared_needs_target_split(self):
        # Backward-compatible path: no split_roles declared anywhere -> provenance alone cannot
        # resolve a held-out split, so an explicit target_split override is still honored.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = self._manifest(
                root, split_names=["train", "train", "train", "test"])
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path])
            self.assertFalse(profile.genuine_holdout_test_available)
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path], target_split="test")
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.resolved_heldout_split, "test")

    def test_conflicting_target_split_and_provenance_role_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = self._manifest(
                root, split_names=["train", "train", "train", "holdout_eval"],
                roles={"train": SPLIT_ROLE_TRAINING,
                       "holdout_eval": SPLIT_ROLE_HELDOUT_EVALUATION})
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path],
                target_split="train")  # conflicts with the declared heldout_eval role
            self.assertFalse(profile.genuine_holdout_test_available)
            self.assertIsNone(profile.genuine_holdout_test_frame_count)
            self.assertIsNone(profile.resolved_heldout_split)

    def test_agreeing_target_split_and_provenance_role_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _sha256 = self._write_db(root, n=4)
            manifest_path = self._manifest(
                root, split_names=["train", "train", "train", "test"],
                roles={"train": SPLIT_ROLE_TRAINING, "test": SPLIT_ROLE_HELDOUT_EVALUATION})
            profile, _ = inspect_teacher_evidence(
                original_training_db_path=db_path,
                split_source_manifest_paths=[manifest_path],
                target_split="test")
            self.assertTrue(profile.genuine_holdout_test_available)
            self.assertEqual(profile.resolved_heldout_split, "test")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
