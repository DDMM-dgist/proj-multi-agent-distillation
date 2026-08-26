"""Canonical-workflow integration tests for Teacher physical validation.

These prove the LIBRARY (validation.teacher_physical_validation) is wired into the canonical
Stage 2 (reference_validation, target establishment) / Stage 7 (train_committee guard) / Stage 11
(physical_validation, Student reproduction) executor path -- not just callable in isolation. Every
observable definition, bin, cutoff, and threshold flows from the ONE frozen Teacher target through
the ONE shared observable dispatcher; the tests assert the wiring never lets the Student redefine,
re-parameterize, or bypass it, and that a workflow which does not opt in is completely unaffected.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

import runtimes.pydantic_ai.executors as ex
from workflow.controller import RunController
from workflow.integrity import artifact_digest


def _grid_frames(n_frames=4, drift=0.02):
    """Alternating Si/O simple-cubic grid in a 10 A periodic box, rigid drift per frame."""
    spacing, n = 2.5, 4
    base, syms = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                base.append([i * spacing, j * spacing, k * spacing])
                syms.append("Si" if (i + j + k) % 2 == 0 else "O")
    base = np.array(base, float)
    frames = []
    for f in range(n_frames):
        pos = base.copy()
        pos[:, 0] += drift * f
        frames.append(Atoms(symbols=syms, positions=pos, cell=[n * spacing] * 3, pbc=True))
    return frames


class _MockTeacherCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    inference_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        type(self).inference_calls += 1
        n = len(atoms)
        self.results["energy"] = -1.0 * n + 1e-3 * float(atoms.get_positions()[:, 0].sum())
        self.results["forces"] = np.zeros((n, 3))


class _MockProvider:
    """Counts calculator construction so INGEST can be proven to build/run no Teacher PES."""
    make_calls = 0

    def make_ase_calculator(self):
        type(self).make_calls += 1
        return _MockTeacherCalc()


def _fake_engine(atoms, calc, params, seed, sample_fn):
    n = int(params["n_steps"])
    stride = int(params["sample_stride"])
    base = atoms.copy()
    for k in range(0, n + 1, stride):
        moved = base.copy()
        moved.positions[:, 0] += 0.02 * k
        moved.calc = calc
        sample_fn(moved, k)


MD_PROTOCOL = {"ensemble": "NVE", "timestep_fs": 1.0, "n_steps": 20, "sample_stride": 2, "seed": 7}
TEACHER_IDENTITY = {"model": "/mock/teacher.pth", "model_sha256": "a" * 64}
RDF_SPEC = {"name": "rdf_SiO", "kind": "rdf_peak_position", "units": "A",
            "params": {"center_species": "Si", "neighbor_species": "O", "r_max": 4.0, "nbins": 200},
            "comparison_criterion": {"operator": "max_abs_deviation", "threshold": 0.5}}
CN_SPEC = {"name": "cn", "kind": "species_coordination", "units": "count",
           "params": {"center_species": "Si", "neighbor_species": "O", "cutoff_A": 3.0}}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        _MockTeacherCalc.inference_calls = 0
        _MockProvider.make_calls = 0
        self.start = os.path.join(self.tmp, "start.xyz")
        write(self.start, _grid_frames()[0])

    def _compute_target(self, target_path=None, traj_path=None, specs=None, profile=None,
                        objective_sha="obj-abc"):
        target_path = target_path or os.path.join(self.tmp, "target.json")
        traj_path = traj_path or os.path.join(self.tmp, "traj.extxyz")
        params = {"objective_profile_sha256": objective_sha, "teacher_identity": TEACHER_IDENTITY,
                  "target_path": target_path, "mode": "COMPUTE",
                  "start_structures_path": self.start, "trajectory_out_path": traj_path,
                  "teacher_calculator_provider": _MockProvider(), "md_engine": _fake_engine}
        if profile is not None:
            params["validation_profile"] = profile
        else:
            params["observable_specs"] = specs if specs is not None else [RDF_SPEC, CN_SPEC]
            params["md_protocol"] = MD_PROTOCOL
        return ex._exec_build_teacher_physical_validation_target({"parameters": params})


class Stage2ComputeTests(_Base):
    def test_1_canonical_stage2_emits_teacher_validation_target(self):
        result = self._compute_target()
        target = result["target"]
        self.assertEqual(target["artifact"], "teacher_validation_target")
        self.assertEqual(target["mode"], "COMPUTE")
        self.assertTrue(target["target_sha256"])
        on_disk = json.loads(open(result["path"]).read())
        self.assertEqual(on_disk["target_sha256"], target["target_sha256"])
        self.assertEqual(result["integrity"]["sha256"], artifact_digest(result["path"])["sha256"])
        # frozen BEFORE any acquisition/training: it is a standalone hash-bound artifact.
        self.assertIn("rdf_SiO", target["observable_status"])

    def test_2_target_sha_registered_in_campaign_state(self):
        wf = {"run_id": "tpv-reg", "inputs": [],
              "stages": [{"name": "reference_validation", "command": None,
                          "outputs": ["artifacts/teacher_validation_target.json"],
                          "gate": {"criteria": ["target frozen"]}}]}
        root = os.path.join(self.tmp, "runroot")
        os.makedirs(root)
        wfp = os.path.join(root, "workflow.yaml")
        yaml.safe_dump(wf, open(wfp, "w"))
        run_dir = os.path.join(root, "run")
        RunController.initialize(wfp, run_dir)
        ctrl = RunController(run_dir)
        target_out = os.path.join(run_dir, "artifacts", "teacher_validation_target.json")
        os.makedirs(os.path.dirname(target_out), exist_ok=True)
        result = self._compute_target(
            target_path=target_out, traj_path=os.path.join(run_dir, "artifacts", "tr.extxyz"))
        ctrl.complete_external_stage("reference_validation", [target_out])
        registered = [a for a in ctrl.state["artifacts"] if a["stage"] == "reference_validation"]
        self.assertTrue(any(a["sha256"] == result["integrity"]["sha256"] for a in registered),
                        "frozen TeacherValidationTarget sha256 was not registered in campaign state")

    def test_8_profile_can_mark_observable_not_applicable(self):
        profile = os.path.join(self.tmp, "vp_obj.yaml")
        yaml.safe_dump({
            "kind": "p", "checks": [],
            "teacher_validation_objectives": ["prohibit_unsupported_generalization_claims"],
            "teacher_physical_validation": {
                "mode": "COMPUTE", "md_protocol": MD_PROTOCOL,
                "observables": [
                    dict(RDF_SPEC, applicable_objectives=[
                        "require_predictive_fidelity_when_evidence_supports_it"]),
                    CN_SPEC,
                ]}}, open(profile, "w"))
        result = self._compute_target(profile=profile)
        status = result["target"]["observable_status"]
        self.assertEqual(status["rdf_SiO"], "NOT_APPLICABLE")
        self.assertEqual(status["cn"], "COMPUTED")


class Stage2IngestTests(_Base):
    def test_7_ingest_mode_performs_zero_teacher_inference(self):
        computed = self._compute_target()
        traj = computed["target"]["trajectory"]["path"]
        traj_sha = computed["target"]["trajectory"]["sha256"]
        _MockTeacherCalc.inference_calls = 0
        _MockProvider.make_calls = 0
        target_path = os.path.join(self.tmp, "ingested.json")
        result = ex._exec_build_teacher_physical_validation_target({"parameters": {
            "objective_profile_sha256": "obj-abc", "teacher_identity": TEACHER_IDENTITY,
            "md_protocol": MD_PROTOCOL, "observable_specs": [RDF_SPEC, CN_SPEC],
            "target_path": target_path, "mode": "INGEST",
            "trajectory_path": traj, "trajectory_sha256": traj_sha,
            # a provider is offered but INGEST must never construct or drive a Teacher calculator:
            "teacher_calculator_provider": _MockProvider()}})
        self.assertEqual(result["target"]["mode"], "INGEST")
        self.assertEqual(result["target"]["ingest_provenance"], "INGESTED_RECOMPUTED")
        self.assertEqual(_MockProvider.make_calls, 0)
        self.assertEqual(_MockTeacherCalc.inference_calls, 0)

    def test_ingest_rejects_trajectory_hash_mismatch(self):
        computed = self._compute_target()
        traj = computed["target"]["trajectory"]["path"]
        with self.assertRaises(ValueError):
            ex._exec_build_teacher_physical_validation_target({"parameters": {
                "objective_profile_sha256": "obj-abc", "teacher_identity": TEACHER_IDENTITY,
                "md_protocol": MD_PROTOCOL, "observable_specs": [RDF_SPEC, CN_SPEC],
                "target_path": os.path.join(self.tmp, "bad.json"), "mode": "INGEST",
                "trajectory_path": traj, "trajectory_sha256": "b" * 64}})


class Stage7GuardTests(_Base):
    def _profile(self, required):
        path = os.path.join(self.tmp, f"vp_{required}.yaml")
        cfg = {"kind": "p", "checks": []}
        if required is not None:
            cfg["teacher_physical_validation"] = {"required": required}
        yaml.safe_dump(cfg, open(path, "w"))
        return path

    def test_3_stage7_cannot_proceed_when_required_target_missing(self):
        profile = self._profile(True)
        trained = {"called": False}
        orig_protect, orig_train = ex._protect_dataset, None
        ex._protect_dataset = lambda *a, **k: None

        def _fake_train(*a, **k):
            trained["called"] = True
            return {}
        import workflow.steps as steps
        orig_train = steps.train_committee
        steps.train_committee = _fake_train
        try:
            with self.assertRaises(ValueError) as ctx:
                ex._exec_train_committee({"parameters": {
                    "dataset": "d", "student_config": "s", "output_dir": "o",
                    "manifest_path": "m", "validation_profile": profile}})
            self.assertIn("REQUIRED", str(ctx.exception))
            self.assertFalse(trained["called"], "training ran despite the missing required target")
        finally:
            ex._protect_dataset = orig_protect
            steps.train_committee = orig_train

    def test_3b_stage7_proceeds_with_valid_bound_target(self):
        profile = self._profile(True)
        target = self._compute_target()["path"]
        # the guard alone must accept a valid, hash-bound target
        ex._assert_teacher_validation_target_bound(
            {"validation_profile": profile, "teacher_validation_target": target})

    def test_3c_stage7_rejects_mutated_target(self):
        profile = self._profile(True)
        target_path = self._compute_target()["path"]
        payload = json.loads(open(target_path).read())
        payload["md_protocol"]["n_steps"] = 999  # tamper after freeze
        open(target_path, "w").write(json.dumps(payload))
        with self.assertRaises(ValueError):
            ex._assert_teacher_validation_target_bound(
                {"validation_profile": profile, "teacher_validation_target": target_path})

    def test_9a_stage7_noop_when_not_required(self):
        # both an explicit required:false and a profile with no block => no-op, even with no target
        ex._assert_teacher_validation_target_bound(
            {"validation_profile": self._profile(False)})
        ex._assert_teacher_validation_target_bound(
            {"validation_profile": self._profile(None)})


class Stage11ReproductionTests(_Base):
    def _profile(self):
        path = os.path.join(self.tmp, "vp11.yaml")
        yaml.safe_dump({"kind": "project-validation", "checks": []}, open(path, "w"))
        return path

    def _student_frames_path(self, traj):
        path = os.path.join(self.tmp, "student.extxyz")
        write(path, read(traj, index=":"))
        return path

    def test_4_stage11_resolves_exact_target_by_sha(self):
        computed = self._compute_target()
        target_path = computed["path"]
        target_sha = computed["target"]["target_sha256"]
        frames_path = self._student_frames_path(computed["target"]["trajectory"]["path"])
        report_path = os.path.join(self.tmp, "report.json")
        # correct sha: resolves and reports it
        result = ex._exec_build_physical_validation_report({"parameters": {
            "validation_profile": self._profile(), "frames_path": frames_path,
            "report_path": report_path, "teacher_validation_target": target_path,
            "teacher_validation_target_sha256": target_sha}})
        self.assertEqual(result["report"]["teacher_validation_target_sha256"], target_sha)
        self.assertEqual(result["teacher_validation_target_sha256"], target_sha)
        # wrong expected sha: refuses to compare against a different target
        with self.assertRaises(ValueError):
            ex._exec_build_physical_validation_report({"parameters": {
                "validation_profile": self._profile(), "frames_path": frames_path,
                "report_path": report_path, "teacher_validation_target": target_path,
                "teacher_validation_target_sha256": "b" * 64}})

    def test_5_stage11_uses_shared_observable_implementation(self):
        # Reproducing the Teacher's OWN trajectory through the same shared kernels must give
        # byte-identical observable values -> zero deviation. Any separate Student implementation
        # would drift. This is the strongest available proof of "same implementation".
        computed = self._compute_target()
        frames_path = self._student_frames_path(computed["target"]["trajectory"]["path"])
        report_path = os.path.join(self.tmp, "report5.json")
        result = ex._exec_build_physical_validation_report({"parameters": {
            "validation_profile": self._profile(), "frames_path": frames_path,
            "report_path": report_path, "teacher_validation_target": computed["path"]}})
        by_name = {c["observable"]: c for c in result["report"]["checks"]}
        self.assertEqual(by_name["rdf_SiO"]["details"]["abs_deviation"], 0.0)
        self.assertEqual(by_name["rdf_SiO"]["status"], "PASS")
        # and the reported student value equals the direct shared-dispatcher evaluation
        from validation.teacher_physical_validation import evaluate_observable
        direct = evaluate_observable(RDF_SPEC, read(frames_path, index=":"))
        self.assertEqual(by_name["rdf_SiO"]["details"]["student_value"], direct["value"])

    def test_6_stage11_rejects_student_side_observable_redefinition(self):
        computed = self._compute_target()
        frames_path = self._student_frames_path(computed["target"]["trajectory"]["path"])
        for bad in ({"r_max": 8.0}, {"nbins": 50}, {"cutoffs": {"Si-O": 2.2}},
                    {"observable_specs": [RDF_SPEC]},
                    {"physical_validation_policy_v2_dict": {"observables": []}}):
            params = {"validation_profile": self._profile(), "frames_path": frames_path,
                      "report_path": os.path.join(self.tmp, "r6.json"),
                      "teacher_validation_target": computed["path"]}
            params.update(bad)
            with self.assertRaises(ValueError) as ctx:
                ex._exec_build_physical_validation_report({"parameters": params})
            self.assertIn("redefine", str(ctx.exception))

    def test_9b_stage11_backward_compatible_without_target(self):
        # A legacy physical_validation profile with real checks and NO bound target must still
        # produce a threshold-bound report exactly as before (feature off = unchanged behavior).
        profile = os.path.join(self.tmp, "vp_legacy.yaml")
        yaml.safe_dump({"kind": "project-validation", "checks": [
            {"name": "rdf_Si_O", "category": "structure", "required": True, "threshold": None},
            {"name": "coordination_Si", "category": "structure", "required": True, "threshold": None},
            {"name": "density", "category": "structure", "required": True, "threshold": None},
        ]}, open(profile, "w"))
        frames_path = os.path.join(self.tmp, "legacy.extxyz")
        write(frames_path, _grid_frames())
        report_path = os.path.join(self.tmp, "legacy_report.json")
        result = ex._exec_build_physical_validation_report({"parameters": {
            "validation_profile": profile, "frames_path": frames_path,
            "report_path": report_path, "r_max": 4.0, "cutoffs": {"Si-O": 3.0, "default": 3.0}}})
        self.assertNotIn("mode", result["report"])  # legacy report shape, no reproduction block
        names = {c["observable"] for c in result["report"]["checks"]}
        self.assertIn("density", names)


if __name__ == "__main__":
    unittest.main()
